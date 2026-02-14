# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tuning script for the fused MoE LoRA Triton kernel.

Sweeps kernel configurations for _fused_moe_lora_shrink and
_fused_moe_lora_expand, benchmarks each, and saves the best config per
batch size to a JSON file that vLLM can load at runtime via
VLLM_TUNED_CONFIG_FOLDER.

Usage examples:

  # Tune w13 shrink + expand for a Mixtral-like gated model (w13_slices=2)
  python benchmarks/kernels/tune_fused_moe_lora.py \
      --hidden-size 4096 --intermediate-size 14336 \
      --lora-rank 16 --max-loras 4 --top-k 2 --num-experts 8 \
      --op-types w13_shrink w13_expand \
      --batch-sizes 1 16 64 256 1024 \
      --dtype torch.bfloat16 --save-dir ./tuned_lora_configs

  # Tune all 4 ops for a non-gated model (e.g. Nemotron, w13_slices=1)
  python benchmarks/kernels/tune_fused_moe_lora.py \
      --hidden-size 2688 --intermediate-size 1856 \
      --lora-rank 16 --max-loras 16 --top-k 2 --num-experts 8 \
      --w13-slices 1 \
      --dtype torch.bfloat16 --save-dir ./tuned_lora_configs

  # Tune all 4 ops for a gated model (default w13_slices=2)
  python benchmarks/kernels/tune_fused_moe_lora.py \
      --hidden-size 4096 --intermediate-size 14336 \
      --lora-rank 16 --max-loras 4 --top-k 2 --num-experts 8 \
      --dtype torch.bfloat16 --save-dir ./tuned_lora_configs

Then set the env var before launching vLLM:
  export VLLM_TUNED_CONFIG_FOLDER=./tuned_lora_configs
"""

import argparse
import gc
import json
import os
import time
from itertools import product

import torch

from vllm import _custom_ops as ops
from vllm.lora.ops.triton_ops.fused_moe_lora_op import (
    _LORA_PTR_DICT,
    _fused_moe_lora_expand,
    _fused_moe_lora_shrink,
)
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.math_utils import next_power_of_2, round_up


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
def get_search_space(
    op_type: str,
    lora_rank: int,
    hidden_size: int,
    intermediate_size: int,
) -> list[dict]:
    """
    Generate a list of kernel config dicts to sweep.  Search ranges are
    informed by production-tuned reference configs (H200 bedrock configs).

    Patterns observed from reference configs:
      - BLOCK_SIZE_M: {16, 32, 64} -- 16 for small batch, 32 for medium,
        64 for large.  128 is never used.
      - Expand: BLOCK_SIZE_N in {64, 128, 256}, BLOCK_SIZE_K = 16,
               split_k = 1 (always).
      - Shrink: BLOCK_SIZE_N in {16, 32}, BLOCK_SIZE_K in {32,64,128,256},
               split_k in {1,2,4,8,16}.
      - num_warps: almost always 4 (occasionally 8 for very small batches).
      - num_stages: 2-5.
      - GROUP_SIZE_M: {1, 4, 8, 16, 32, 64}.
    """
    block_m_range = [16, 32, 64]
    num_warps_range = [4, 8]
    group_m_range = [1, 4, 8, 16, 32, 64]
    num_stages_range = [2, 3, 4, 5]
    split_k_range = [1]  # default for expand

    rank_po2 = next_power_of_2(lora_rank)

    if op_type.endswith("_shrink"):
        # Shrink: GEMM is (M, K=input_dim) x (K, N=rank) -> (M, N)
        # Reference shows BLOCK_SIZE_N in {16, 32}, capped at rank
        block_n_range = [v for v in [16, 32] if v <= rank_po2]
        if not block_n_range:
            block_n_range = [rank_po2]
        # Reference shows large BLOCK_SIZE_K for shrink
        block_k_range = [32, 64, 128, 256]
        # Reference shows aggressive split_k for small batches
        split_k_range = [1, 2, 4, 8, 16]
    else:
        # Expand: GEMM is (M, K=rank) x (K, N=output_dim) -> (M, N)
        # Reference shows BLOCK_SIZE_K = 16 (small, capped at rank)
        block_k_range = [v for v in [16, 32] if v <= rank_po2]
        if not block_k_range:
            block_k_range = [max(16, rank_po2)]
        # Reference shows large BLOCK_SIZE_N for expand
        block_n_range = [64, 128, 256]

    configs = []
    for bm, bn, bk, gm, nw, ns, sk in product(
        block_m_range,
        block_n_range,
        block_k_range,
        group_m_range,
        num_warps_range,
        num_stages_range,
        split_k_range,
    ):
        # Basic validity checks
        if bm * bn < 64:
            continue
        # Shared memory heuristic (2 bytes per element for bf16/fp16)
        lds = bk * bm * 2 + bk * bn * 2
        if lds > 65536:
            continue
        # For shrink with large split_k, ensure K dimension is divisible
        # by (split_k * block_k).  Determine the K dimension.
        if op_type.endswith("_shrink") and sk > 1:
            if "w13" in op_type:
                k_dim = hidden_size
            else:
                k_dim = intermediate_size
            if k_dim % (sk * bk) != 0:
                continue
        configs.append({
            "block_m": bm,
            "block_n": bn,
            "block_k": bk,
            "group_size_m": gm,
            "num_warps": nw,
            "num_stages": ns,
            "split_k": sk,
        })

    print(f"  Search space for {op_type}: {len(configs)} configs")
    return configs


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_moe_lora_data(
    block_size_m: int,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    max_loras: int,
    token_lora_mapping: torch.Tensor,
    adapter_enabled: torch.Tensor,
    lora_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create topk_ids, run moe_lora_align_block_size, return
    (topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded).
    """
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32
    )
    topk_weights = torch.rand(
        (num_tokens, top_k), device=device, dtype=torch.float32
    )

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size_m - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size_m)

    sorted_ids = torch.empty(
        (max_loras * max_num_tokens_padded,),
        dtype=torch.int32,
        device=device,
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size_m)
    expert_ids = torch.empty(
        (max_loras * max_num_m_blocks,),
        dtype=torch.int32,
        device=device,
    )
    num_tokens_post_pad = torch.empty(
        max_loras, dtype=torch.int32, device=device
    )

    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size_m,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
    )

    sorted_token_ids = sorted_ids.view(max_loras, -1)
    expert_ids = expert_ids.view(max_loras, -1)
    return topk_weights, sorted_token_ids, expert_ids, num_tokens_post_pad


def make_tensors(
    batch_size: int,
    shrink_input_dim: int,
    expand_output_dim: int,
    lora_rank: int,
    max_loras: int,
    num_experts: int,
    top_k: int,
    num_slices: int,
    mul_routed_weight: bool,
    dtype: torch.dtype,
    device: torch.device,
):
    """Allocate all synthetic tensors needed for both shrink and expand.

    Args:
        shrink_input_dim: K for the shrink GEMM.
            w13 ops: hidden_size, w2 ops: intermediate_size.
        expand_output_dim: N for the expand GEMM.
            w13 ops: intermediate_size, w2 ops: hidden_size.
        mul_routed_weight: True for w2 (down) ops. When True, the shrink
            input is already expanded by top_k (each token repeated top_k
            times), so the input has M * top_k rows instead of M.
    """
    M = batch_size
    num_tokens = M * top_k

    # For w2 (down) ops the input is the intermediate activation that has
    # already been expanded by top_k.  token_mapping_factor=1 in the
    # kernel so it indexes rows 0..M*top_k-1 directly.
    # For w13 (gate_up) ops the input has M rows and the kernel divides
    # the token index by top_k (token_mapping_factor=top_k).
    shrink_input_rows = M * top_k if mul_routed_weight else M

    # Shrink input
    qcurr_hidden_states = torch.randn(
        (shrink_input_rows, shrink_input_dim), dtype=dtype, device=device
    )

    # LoRA A weights: (max_loras, num_experts, lora_rank, shrink_input_dim)
    lora_a_stacked = [
        torch.randn(
            (max_loras, num_experts, lora_rank, shrink_input_dim),
            dtype=dtype,
            device=device,
        )
        for _ in range(num_slices)
    ]

    # LoRA B weights: (max_loras, num_experts, expand_output_dim, lora_rank)
    lora_b_stacked = [
        torch.randn(
            (max_loras, num_experts, expand_output_dim, lora_rank),
            dtype=dtype,
            device=device,
        )
        for _ in range(num_slices)
    ]

    # Shrink output: (num_slices, M, top_k, lora_rank)
    a_intermediate_cache = torch.zeros(
        (num_slices, M, top_k, lora_rank),
        dtype=dtype,
        device=device,
    )

    # Expand output: (M, top_k, expand_output_dim * num_slices)
    expand_output = torch.zeros(
        (M, top_k, expand_output_dim * num_slices),
        dtype=dtype,
        device=device,
    )

    # Metadata — the C++ op expects int32 for all index tensors
    token_lora_mapping = torch.randint(
        0, max_loras, (M,), device=device, dtype=torch.int32
    )
    adapter_enabled = torch.ones(
        max_loras + 1, dtype=torch.int32, device=device
    )
    # Build lora_ids from unique values in token_lora_mapping
    unique_ids = token_lora_mapping.unique()
    lora_ids = torch.full(
        (max_loras,), -1, dtype=torch.int32, device=device
    )
    lora_ids[: unique_ids.shape[0]] = unique_ids
    num_active_loras = unique_ids.shape[0]

    return {
        "qcurr_hidden_states": qcurr_hidden_states,
        "lora_a_stacked": lora_a_stacked,
        "lora_b_stacked": lora_b_stacked,
        "a_intermediate_cache": a_intermediate_cache,
        "expand_output": expand_output,
        "token_lora_mapping": token_lora_mapping,
        "adapter_enabled": adapter_enabled,
        "lora_ids": lora_ids,
        "num_active_loras": num_active_loras,
        "M": M,
        "num_tokens": num_tokens,
    }


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------
def benchmark_shrink(
    tensors: dict,
    config: dict,
    top_k: int,
    num_experts: int,
    max_loras: int,
    lora_rank: int,
    num_slices: int,
    mul_routed_weight: bool,
    num_warmup: int = 3,
    num_iters: int = 20,
) -> float:
    """
    Benchmark _fused_moe_lora_shrink with the given config.
    Returns latency in microseconds, or float('inf') on failure.
    """
    device = tensors["qcurr_hidden_states"].device
    block_size_m = config["block_m"]
    M = tensors["M"]

    # Prepare block-aligned data for this BLOCK_SIZE_M
    topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded = (
        prepare_moe_lora_data(
            block_size_m=block_size_m,
            num_tokens=M,
            top_k=top_k,
            num_experts=num_experts,
            max_loras=max_loras,
            token_lora_mapping=tensors["token_lora_mapping"],
            adapter_enabled=tensors["adapter_enabled"],
            lora_ids=tensors["lora_ids"],
            device=device,
        )
    )

    EM = sorted_token_ids.shape[1]
    num_tokens = tensors["num_tokens"]
    N = lora_rank
    K = tensors["qcurr_hidden_states"].shape[1]  # shrink_input_dim

    # Reset output
    a_out = tensors["a_intermediate_cache"].zero_()

    kwargs = dict(
        a_intermediate_cache1=a_out,
        qcurr_hidden_states=tensors["qcurr_hidden_states"],
        lora_a_stacked=tensors["lora_a_stacked"],
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        token_lora_mapping=tensors["token_lora_mapping"],
        top_k_num=top_k,
        lora_ids=tensors["lora_ids"],
        adapter_enabled=tensors["adapter_enabled"],
        device=device,
        N=N,
        M=M,
        EM=EM,
        K=K,
        num_tokens=num_tokens,
        num_experts=num_experts,
        num_slices=num_slices,
        block_size_m=config["block_m"],
        block_size_n=config["block_n"],
        block_size_k=config["block_k"],
        group_size_m=config["group_size_m"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        split_k=config["split_k"],
        num_active_loras=tensors["num_active_loras"],
        mul_routed_weight=mul_routed_weight,
        use_gdc=False,
    )

    try:
        # Warmup (also triggers JIT compilation)
        for _ in range(num_warmup):
            _fused_moe_lora_shrink(**kwargs)
        torch.cuda.synchronize()

        # Benchmark with CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            _fused_moe_lora_shrink(**kwargs)
        end_event.record()
        end_event.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        return elapsed_ms / num_iters * 1000  # convert to microseconds
    except Exception as e:
        print(f"    [skip] shrink config {config}: {e}")
        # A CUDA error (e.g. illegal memory access) corrupts the context.
        # Synchronize to clear the error state before the next config.
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return float("inf")


def benchmark_expand(
    tensors: dict,
    config: dict,
    top_k: int,
    num_experts: int,
    max_loras: int,
    lora_rank: int,
    num_slices: int,
    mul_routed_weight: bool,
    num_warmup: int = 3,
    num_iters: int = 20,
) -> float:
    """
    Benchmark _fused_moe_lora_expand with the given config.
    Returns latency in microseconds, or float('inf') on failure.
    """
    device = tensors["qcurr_hidden_states"].device
    block_size_m = config["block_m"]
    M = tensors["M"]

    # Prepare block-aligned data for this BLOCK_SIZE_M
    topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded = (
        prepare_moe_lora_data(
            block_size_m=block_size_m,
            num_tokens=M,
            top_k=top_k,
            num_experts=num_experts,
            max_loras=max_loras,
            token_lora_mapping=tensors["token_lora_mapping"],
            adapter_enabled=tensors["adapter_enabled"],
            lora_ids=tensors["lora_ids"],
            device=device,
        )
    )

    EM = sorted_token_ids.shape[1]
    num_tokens = tensors["num_tokens"]

    # The expand function internally reassigns N and K:
    #   K = max_lora_rank
    #   N = w1_output_dim_size = lora_b_stacked[0].shape[2]
    w1_output_dim_size = tensors["lora_b_stacked"][0].shape[2]

    # Use the shrink output as expand input (fill with random data)
    a_intermediate = tensors["a_intermediate_cache"].clone()
    a_intermediate.normal_()

    # Reset output
    expand_out = tensors["expand_output"].zero_()

    kwargs = dict(
        output=expand_out,
        a_intermediate_cache1=a_intermediate,
        lora_b_stacked=tensors["lora_b_stacked"],
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        token_lora_mapping=tensors["token_lora_mapping"],
        top_k_num=top_k,
        lora_ids=tensors["lora_ids"],
        adapter_enabled=tensors["adapter_enabled"],
        device=device,
        N=lora_rank,  # overwritten inside _fused_moe_lora_expand
        M=M,
        EM=EM,
        K=tensors["qcurr_hidden_states"].shape[1],  # overwritten inside
        num_tokens=num_tokens,
        num_experts=num_experts,
        num_slices=num_slices,
        max_lora_rank=lora_rank,
        w1_output_dim_size=w1_output_dim_size,
        block_size_m=config["block_m"],
        block_size_n=config["block_n"],
        block_size_k=config["block_k"],
        group_size_m=config["group_size_m"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        split_k=config["split_k"],
        num_active_loras=tensors["num_active_loras"],
        mul_routed_weight=mul_routed_weight,
        offset=0,
        use_gdc=False,
    )

    try:
        for _ in range(num_warmup):
            _fused_moe_lora_expand(**kwargs)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            _fused_moe_lora_expand(**kwargs)
        end_event.record()
        end_event.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        return elapsed_ms / num_iters * 1000
    except Exception as e:
        print(f"    [skip] expand config {config}: {e}")
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return float("inf")


# ---------------------------------------------------------------------------
# Tuning loop
# ---------------------------------------------------------------------------
def tune_op(
    op_type: str,
    batch_sizes: list[int],
    hidden_size: int,
    intermediate_size: int,
    lora_rank: int,
    max_loras: int,
    top_k: int,
    num_experts: int,
    dtype: torch.dtype,
    num_iters: int,
    w13_slices: int = 2,
    fixed_block_m_map: dict[int, int] | None = None,
) -> dict[int, dict]:
    """
    For each batch size, sweep all configs and return the best one.
    Returns {batch_size: best_config_dict}.

    If *fixed_block_m_map* is provided, the search space for each batch size
    is filtered to only include configs whose ``block_m`` matches the value
    in the map.  This is used to constrain the shrink op to use the same
    BLOCK_SIZE_M as the expand op that was already tuned.
    """
    is_shrink = op_type.endswith("_shrink")
    is_w13 = "w13" in op_type
    num_slices = w13_slices if is_w13 else 1
    mul_routed_weight = not is_w13  # w2 ops multiply by routed weight

    # w13 (gate_up): shrink K=hidden_size, expand N=intermediate_size
    # w2  (down):    shrink K=intermediate_size, expand N=hidden_size
    if is_w13:
        shrink_input_dim = hidden_size
        expand_output_dim = intermediate_size
    else:
        shrink_input_dim = intermediate_size
        expand_output_dim = hidden_size

    full_search_space = get_search_space(
        op_type, lora_rank, hidden_size, intermediate_size
    )

    device = torch.device("cuda")
    best_configs: dict[int, dict] = {}

    for batch_size in batch_sizes:
        # Optionally constrain BLOCK_SIZE_M for this batch size
        if fixed_block_m_map is not None and batch_size in fixed_block_m_map:
            required_bm = fixed_block_m_map[batch_size]
            search_space = [
                c for c in full_search_space if c["block_m"] == required_bm
            ]
            print(
                f"\n  Tuning {op_type} | batch_size={batch_size} "
                f"(BLOCK_SIZE_M fixed to {required_bm}, "
                f"{len(search_space)} configs)"
            )
        else:
            search_space = full_search_space
            print(f"\n  Tuning {op_type} | batch_size={batch_size}")

        # Allocate tensors once per batch size
        tensors = make_tensors(
            batch_size=batch_size,
            shrink_input_dim=shrink_input_dim,
            expand_output_dim=expand_output_dim,
            lora_rank=lora_rank,
            max_loras=max_loras,
            num_experts=num_experts,
            top_k=top_k,
            num_slices=num_slices,
            mul_routed_weight=mul_routed_weight,
            dtype=dtype,
            device=device,
        )

        best_time = float("inf")
        best_config = None
        num_tried = 0

        for i, config in enumerate(search_space):
            # Clear cached Triton pointers to avoid stale data
            _LORA_PTR_DICT.clear()

            if is_shrink:
                t = benchmark_shrink(
                    tensors=tensors,
                    config=config,
                    top_k=top_k,
                    num_experts=num_experts,
                    max_loras=max_loras,
                    lora_rank=lora_rank,
                    num_slices=num_slices,
                    mul_routed_weight=mul_routed_weight,
                    num_iters=num_iters,
                )
            else:
                t = benchmark_expand(
                    tensors=tensors,
                    config=config,
                    top_k=top_k,
                    num_experts=num_experts,
                    max_loras=max_loras,
                    lora_rank=lora_rank,
                    num_slices=num_slices,
                    mul_routed_weight=mul_routed_weight,
                    num_iters=num_iters,
                )

            num_tried += 1
            if t < best_time:
                best_time = t
                best_config = config.copy()

            # Periodic progress
            if (i + 1) % 50 == 0 or i == len(search_space) - 1:
                print(
                    f"    [{i+1}/{len(search_space)}] "
                    f"best so far: {best_time:.1f} us "
                    f"config={best_config}"
                )

            # Periodic GC to avoid OOM during long sweeps
            if (i + 1) % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        print(
            f"  => batch_size={batch_size}: best={best_time:.1f} us "
            f"({num_tried} configs tried)"
        )
        print(f"     config={best_config}")
        best_configs[batch_size] = best_config

        # Free tensors between batch sizes
        del tensors
        gc.collect()
        torch.cuda.empty_cache()

    return best_configs


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def save_configs(
    best_configs: dict[int, dict],
    op_type: str,
    max_loras: int,
    num_slices: int,
    lora_rank: int,
    hidden_size: int,
    intermediate_size: int,
    save_dir: str,
) -> str:
    """
    Save tuned configs to a JSON file matching the format expected by
    vllm.lora.ops.triton_ops.utils.get_lora_op_configs().

    Nested structure:
      config[max_loras][num_slices][M][k][n][moe_intermediate_size] = {...}

    Where k=rank, n=hidden_size for fused_moe_lora ops (see
    get_lora_op_configs line 263: for non-"shrink" op_types,
    k=rank, n=hidden_size).
    """
    loras_key = str(max_loras)
    slices_key = str(num_slices)
    k_key = str(lora_rank)
    n_key = str(hidden_size)
    i_key = str(intermediate_size)

    result = {loras_key: {slices_key: {}}}

    for batch_size, config in best_configs.items():
        m_key = str(batch_size)
        result[loras_key][slices_key][m_key] = {
            k_key: {n_key: {i_key: config}}
        }

    # Map op_type to the filename op_type string used by load_lora_op_config
    # e.g. "w13_shrink" -> "fused_moe_lora_w13_shrink"
    file_op_type = f"fused_moe_lora_{op_type}"

    gpu_name = torch.cuda.get_device_name()
    gpu_name = gpu_name.replace(" ", "_").replace("-", "_")
    filename = f"{gpu_name}_{file_op_type.upper()}.json"

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    # If the file already exists, merge the new configs into it
    # (supports tuning different max_loras / num_slices combinations
    # incrementally).
    if os.path.exists(filepath):
        with open(filepath) as f:
            existing = json.load(f)
        # Deep merge
        if loras_key not in existing:
            existing[loras_key] = {}
        if slices_key not in existing[loras_key]:
            existing[loras_key][slices_key] = {}
        existing[loras_key][slices_key].update(
            result[loras_key][slices_key]
        )
        result = existing

    print(f"\nWriting tuned configs to {filepath}")
    with open(filepath, "w") as f:
        json.dump(result, f, indent=4)
        f.write("\n")

    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_OP_TYPES = ["w13_shrink", "w13_expand", "w2_shrink", "w2_expand"]


def _tune_and_save(
    op_type: str,
    args: argparse.Namespace,
    fixed_block_m_map: dict[int, int] | None = None,
) -> dict[int, dict]:
    """Tune a single op_type and save results.  Returns best_configs."""
    is_w13 = "w13" in op_type
    num_slices = args.w13_slices if is_w13 else 1

    print(f"\n{'='*60}")
    print(f"Tuning: {op_type}  (num_slices={num_slices})")
    if fixed_block_m_map:
        print(
            f"  BLOCK_SIZE_M constrained by expand results: "
            f"{fixed_block_m_map}"
        )
    print(f"{'='*60}")

    best_configs = tune_op(
        op_type=op_type,
        batch_sizes=args.batch_sizes,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        lora_rank=args.lora_rank,
        max_loras=args.max_loras,
        top_k=args.top_k,
        num_experts=args.num_experts,
        dtype=args.dtype,
        num_iters=args.num_iters,
        w13_slices=args.w13_slices,
        fixed_block_m_map=fixed_block_m_map,
    )

    filepath = save_configs(
        best_configs=best_configs,
        op_type=op_type,
        max_loras=args.max_loras,
        num_slices=num_slices,
        lora_rank=args.lora_rank,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        save_dir=args.save_dir,
    )

    print(f"\n  Summary for {op_type}:")
    for bs, cfg in best_configs.items():
        print(f"    batch_size={bs:>5d}: {cfg}")
    print(f"  Saved to: {filepath}")

    return best_configs


def main(args: argparse.Namespace):
    print("=" * 60)
    print("Fused MoE LoRA Kernel Tuning")
    print("=" * 60)
    print(f"  GPU           : {torch.cuda.get_device_name()}")
    print(f"  hidden_size   : {args.hidden_size}")
    print(f"  intermediate  : {args.intermediate_size}")
    print(f"  lora_rank     : {args.lora_rank}")
    print(f"  max_loras     : {args.max_loras}")
    print(f"  top_k         : {args.top_k}")
    print(f"  num_experts   : {args.num_experts}")
    print(f"  w13_slices    : {args.w13_slices}")
    print(f"  dtype         : {args.dtype}")
    print(f"  batch_sizes   : {args.batch_sizes}")
    print(f"  op_types      : {args.op_types}")
    print(f"  num_iters     : {args.num_iters}")
    print(f"  save_dir      : {args.save_dir}")
    print()

    overall_start = time.time()

    requested = set(args.op_types)

    # Determine which (expand, shrink) pairs are requested.
    # Expand is always tuned first to determine BLOCK_SIZE_M, which is then
    # fixed for the corresponding shrink op so that both use the same token
    # alignment.
    for prefix in ("w13", "w2"):
        expand_op = f"{prefix}_expand"
        shrink_op = f"{prefix}_shrink"

        has_expand = expand_op in requested
        has_shrink = shrink_op in requested

        if not has_expand and not has_shrink:
            continue

        # --- 1. Tune expand first (bottleneck) ---
        if has_expand:
            expand_best = _tune_and_save(expand_op, args)
        else:
            expand_best = None

        # --- 2. Tune shrink with BLOCK_SIZE_M fixed from expand ---
        if has_shrink:
            fixed_block_m_map = None
            if expand_best is not None:
                # Extract the BLOCK_SIZE_M chosen by expand for each
                # batch size and use it to constrain shrink.
                fixed_block_m_map = {
                    bs: cfg["block_m"]
                    for bs, cfg in expand_best.items()
                }
                print(
                    f"\n  Constraining {shrink_op} BLOCK_SIZE_M to match "
                    f"{expand_op} results"
                )
            _tune_and_save(shrink_op, args, fixed_block_m_map)

    elapsed = time.time() - overall_start
    print(f"\nTotal tuning time: {elapsed:.1f} seconds")
    print(
        f"\nTo use these configs, set:\n"
        f"  export VLLM_TUNED_CONFIG_FOLDER={os.path.abspath(args.save_dir)}"
    )


if __name__ == "__main__":

    def to_torch_dtype(dt: str) -> torch.dtype:
        mapping = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dt not in mapping:
            raise ValueError(
                f"Unsupported dtype '{dt}'. "
                f"Choose from: {list(mapping.keys())}"
            )
        return mapping[dt]

    parser = FlexibleArgumentParser(
        description="Tune fused MoE LoRA Triton kernel configs",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        required=True,
        help="Model hidden size (e.g. 4096)",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        required=True,
        help=(
            "MoE intermediate size per partition (e.g. 14336).\n"
            "This is the per-expert FFN intermediate dim after TP sharding."
        ),
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        required=True,
        help="LoRA rank (e.g. 16)",
    )
    parser.add_argument(
        "--max-loras",
        type=int,
        default=4,
        help="Max number of LoRA adapters (default: 4)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of experts activated per token (default: 2)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=8,
        help="Total number of MoE experts (default: 8)",
    )
    parser.add_argument(
        "--w13-slices",
        type=int,
        default=2,
        choices=[1, 2],
        help=(
            "Number of slices for w13 (gate/up) ops (default: 2).\n"
            "  2 = gated MoE with act_and_mul (gate + up, e.g. Mixtral)\n"
            "  1 = non-gated MoE (single projection, e.g. Nemotron)"
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        help="Batch sizes (num_tokens) to tune for",
    )
    parser.add_argument(
        "--op-types",
        nargs="+",
        type=str,
        default=ALL_OP_TYPES,
        choices=ALL_OP_TYPES,
        help=(
            "Which ops to tune. Choices:\n"
            "  w13_shrink  - gate/up LoRA A (hidden -> rank)\n"
            "  w13_expand  - gate/up LoRA B (rank -> intermediate)\n"
            "  w2_shrink   - down LoRA A (intermediate -> rank)\n"
            "  w2_expand   - down LoRA B (rank -> hidden)"
        ),
    )
    parser.add_argument(
        "--dtype",
        type=to_torch_dtype,
        default=torch.bfloat16,
        help="Data type (default: torch.bfloat16)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=20,
        help="Number of benchmark iterations per config (default: 20)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./tuned_lora_configs",
        help="Directory to save tuned config JSON files (default: ./tuned_lora_configs)",
    )

    args = parser.parse_args()
    main(args)
