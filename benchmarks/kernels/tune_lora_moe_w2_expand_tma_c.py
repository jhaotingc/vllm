# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tune and benchmark W2 expand kernel with TMA C descriptor.

Compares a custom expand kernel that uses TMA descriptors for the
output (C) load+add+store against the baseline pointer-based expand.
The TMA-C variant writes to a contiguous sorted-order buffer (which
would need an unpermute in production, not included here).

The hypothesis: TMA C reduces register pressure by offloading 2D
address calculation to the TMA hardware unit, improving occupancy
and performance at larger batch sizes.

Usage:

  # Tune TMA-C + compare with baseline
  python tune_lora_moe_w2_expand_tma_c.py \\
      --hidden-size 3072 --intermediate-size 2944 \\
      --lora-rank 32 --max-loras 8 --top-k 4 --num-experts 32 \\
      --batch-sizes 64 256 512 1024 2048

  # TMA-C only (skip baseline comparison)
  python tune_lora_moe_w2_expand_tma_c.py \\
      --hidden-size 3072 --intermediate-size 2944 \\
      --lora-rank 32 --max-loras 8 --top-k 4 --num-experts 32 \\
      --batch-sizes 64 256 512 1024 2048 --tma-only
"""

import contextlib
import gc
import os
import time

import torch

from vllm.lora.ops.triton_ops.fused_moe_lora_op import (
    _adjust_kernel_inputs,
    _fused_moe_lora_expand,
    _get_expert_id,
    _get_lora_id,
    _get_ptr,
    _get_token_offs,
    _LORA_PTR_DICT,
)
from vllm.lora.ops.triton_ops.utils import (
    _LORA_A_PTR_DICT,
    _LORA_B_PTR_DICT,
    supports_pdl,
    supports_tma,
)
from vllm.triton_utils import tl, triton
from vllm.triton_utils.allocation import set_triton_allocator
from vllm.utils.argparse_utils import FlexibleArgumentParser

with contextlib.suppress(ImportError):
    import triton.tools.tensor_descriptor  # noqa: F401
    from triton.tools.tensor_descriptor import TensorDescriptor

# Shared utilities from the main tuner
from tune_lora_moe import (
    _clear_caches,
    _get_search_space,
    _load_config,
    _make_tensors,
    _prepare_moe_data,
    _save_configs,
    _should_use_naive,
    _timed,
)


# =====================================================================
# TMA-C expand kernel
# =====================================================================
# This is a modified version of vllm's _fused_moe_lora_kernel where
# the C (output) tile uses a TMA descriptor for load+add+store instead
# of pointer-based access. A and B paths are identical to the baseline.
#
# Key difference:
#   Baseline:  c_ptrs = cur_c_ptr + offs_token[:, None] * stride_cm + ...
#              prev = tl.load(c_ptrs, mask=c_mask, other=0.0)
#              tl.store(c_ptrs, prev + accumulator, mask=c_mask)
#
#   TMA-C:     offs_cm = lora_id * EM + pid_m * BLOCK_SIZE_M
#              offs_cn = pid_n * BLOCK_SIZE_N
#              prev = c_desc.load([offs_cm, offs_cn])
#              c_desc.store([offs_cm, offs_cn], prev + accumulator)
#
# The TMA unit handles the full 2D address calculation in hardware,
# replacing ~BLOCK_M*BLOCK_N address registers with 2 scalar offsets.
# =====================================================================


@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "stride_tl",
        "stride_el",
        "slice_a_size",
    ]
)
def _expand_tma_c_kernel(
    # A input (intermediate cache from shrink)
    a_ptr,
    a_desc,
    # B weights (lora_b_stacked pointer tensor)
    b_ptr,
    b_desc,
    # C output - TMA descriptor for contiguous sorted-order buffer
    c_desc,
    c_ptr,  # dummy pointer for dtype extraction only
    # Metadata
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    token_lora_mapping_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    top_k_num,
    lora_ids,
    adapter_enabled,
    max_loras,
    # Strides (A)
    stride_am,
    stride_ak,
    # Strides (B)
    stride_bl,
    stride_be,
    stride_bk,
    stride_bn,
    # Strides (routing metadata)
    stride_tl,
    stride_el,
    slice_a_size,
    # Constexpr config
    num_slice_a: tl.constexpr,
    token_mapping_factor: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    """Expand kernel with TMA C load+add+store.

    C is a contiguous 2D buffer in sorted token order: (max_loras * EM, N).
    Uses c_desc for TMA load and store, eliminating pointer arithmetic
    for the output tile and reducing register pressure.
    """
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    grid_k = tl.cdiv(K, BLOCK_SIZE_K)

    # Grouped ordering for L2 locality
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    # ── Lora / expert / token setup (same as baseline) ────────────
    lora_id = _get_lora_id(
        lora_ids, token_lora_mapping_ptr, lora_idx, pid_m,
        top_k_num, naive_block_assignment,
    )
    if lora_id == -1:
        return
    if tl.load(adapter_enabled + lora_id) == 0:
        return
    if lora_id >= max_loras:
        return

    if not naive_block_assignment:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

    expert_id = _get_expert_id(
        expert_ids_ptr, lora_id, pid_m, stride_el,
        max_loras, naive_block_assignment,
    )
    if expert_id == -1:
        return

    # Token offsets (still needed for routing weight lookup)
    offs_token = _get_token_offs(
        sorted_token_ids_ptr, lora_id, pid_m, offs, stride_tl,
        max_loras, num_valid_tokens, naive_block_assignment, BLOCK_SIZE_M,
    )
    token_mask = offs_token < num_valid_tokens

    # ── A setup (identical to baseline expand path) ───────────────
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size

    if USE_TMA and a_desc is not None:
        offs_am = (
            slice_id * max_loras * EM
            + lora_id * EM
            + pid_m * BLOCK_SIZE_M // token_mapping_factor
        )
        offs_ak = 0
    else:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = cur_a_ptr + (
            offs_token[:, None] // token_mapping_factor * stride_am
            + offs_k[None, :] * stride_ak
        )

    # ── B setup (identical to baseline) ───────────────────────────
    cur_b_ptr = tl.load(b_ptr + slice_id).to(
        tl.pointer_type(c_ptr.dtype.element_ty)
    )

    if USE_TMA:
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_bk = 0
        if b_desc is None:
            b_desc = tl.make_tensor_descriptor(
                cur_b_ptr,
                shape=[max_loras, num_experts, N, K],
                strides=[stride_bl, stride_be, stride_bn, stride_bk],
                block_shape=[1, 1, BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
    else:
        offs_k_b = tl.arange(0, BLOCK_SIZE_K)
        offs_bn_ptr = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(
            tl.int32
        )
        b_ptrs = (
            cur_b_ptr
            + lora_id * stride_bl
            + expert_id * stride_be
            + offs_k_b[:, None] * stride_bk
            + offs_bn_ptr[None, :] * stride_bn
        )

    # ── Main GEMM loop (identical to baseline) ────────────────────
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, grid_k):
        cur_k_offset = k * BLOCK_SIZE_K

        # Load B
        if b_desc is not None:
            b = (
                b_desc.load(
                    [lora_id, expert_id, offs_bn, offs_bk + cur_k_offset]
                )
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .T
            )
        else:
            k_remaining = K - cur_k_offset
            b_mask = (offs_k_b[:, None] < k_remaining) & (
                offs_bn_ptr[None, :] < N
            )
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # Load A
        if a_desc is not None:
            a = a_desc.load([offs_am, offs_ak + cur_k_offset])
        else:
            k_remaining = K - cur_k_offset
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            a_ptrs += BLOCK_SIZE_K * stride_ak

        accumulator += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))

    # ── Apply routing weights (always True for w2) ────────────────
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token, mask=token_mask, other=0.0
        )
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # ── C: TMA load + add + store (THE KEY CHANGE) ───────────────
    # Instead of computing scattered c_ptrs from offs_token and doing
    # masked pointer load/store, we use TMA with simple 2D offsets
    # into the contiguous sorted-order buffer.
    offs_cm = lora_id * EM + pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    prev = c_desc.load([offs_cm, offs_cn])
    c_desc.store([offs_cm, offs_cn], prev + accumulator)


# =====================================================================
# Wrapper: launch TMA-C expand kernel
# =====================================================================
@torch.inference_mode()
def _expand_tma_c(
    c_sorted,  # contiguous (max_loras * EM, N) output buffer
    a_intermediate_cache1,
    lora_b_stacked,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    token_lora_mapping,
    top_k_num,
    lora_ids,
    adapter_enabled,
    device,
    N,
    M,
    EM,
    K,
    num_tokens,
    num_experts,
    num_slices,
    max_lora_rank,
    w1_output_dim_size,
    block_size_m,
    block_size_n,
    block_size_k,
    group_size_m,
    num_warps,
    num_stages,
    num_active_loras,
    mul_routed_weight,
):
    """Launch the TMA-C expand kernel."""
    assert sorted_token_ids is not None, "TMA-C requires sorted routing"

    b_ptr = _get_ptr(lora_b_stacked, device)
    K_actual = max_lora_rank
    N_actual = w1_output_dim_size
    w1_lora_b_stacked = lora_b_stacked[0]

    a_flat = a_intermediate_cache1.view(-1, a_intermediate_cache1.shape[-1])

    grid_lora_dim, stride_tl, stride_el = _adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N_actual, META["BLOCK_SIZE_N"]),
        len(lora_b_stacked),
        grid_lora_dim,
    )

    # Create TMA descriptors
    a_desc = TensorDescriptor.from_tensor(
        a_flat, [block_size_m, block_size_k]
    )
    b_desc = TensorDescriptor.from_tensor(
        lora_b_stacked[0], [1, 1, block_size_n, block_size_k]
    )
    c_desc = TensorDescriptor.from_tensor(
        c_sorted, [block_size_m, block_size_n]
    )

    _expand_tma_c_kernel[grid](
        a_flat,
        a_desc,
        b_ptr,
        b_desc,
        c_desc,
        c_sorted,  # for dtype extraction
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        N_actual,
        K_actual,
        EM,
        num_tokens,
        num_experts,
        top_k_num,
        lora_ids,
        adapter_enabled,
        lora_b_stacked[0].shape[0],  # max_loras
        a_flat.stride(0),
        a_flat.stride(1),
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(1),
        w1_lora_b_stacked.stride(3),
        w1_lora_b_stacked.stride(2),
        stride_tl,
        stride_el,
        slice_a_size=a_flat.numel() // num_slices,
        num_slice_a=num_slices,
        token_mapping_factor=1,
        naive_block_assignment=False,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
        USE_TMA=True,
    )


# =====================================================================
# Benchmark functions
# =====================================================================
def _bench_tma_c(
    tensors,
    expand_cfg,
    top_k,
    num_experts,
    max_loras,
    lora_rank,
    num_slices,
    block_size_m,
    num_iters,
    device,
    mul_routed_weight=False,
):
    """Benchmark the TMA-C expand kernel. Returns us."""
    M = tensors["M"]
    assert not _should_use_naive(M, top_k, num_experts, max_loras), (
        "TMA-C requires sorted routing (non-naive). "
        f"bs={M} is too small for these params."
    )

    topk_weights, sorted_token_ids, expert_ids, ntpp = _prepare_moe_data(
        tensors, block_size_m, top_k, num_experts, max_loras, device
    )
    EM = sorted_token_ids.shape[1]
    num_loras = sorted_token_ids.shape[0]
    w1_output_dim_size = tensors["lora_b"][0].shape[2]

    # Intermediate cache in sorted order (expand input)
    cache_shape = (num_slices, num_loras, EM, lora_rank)
    a_intermediate = torch.randn(
        cache_shape, dtype=torch.bfloat16, device=device
    )

    # Contiguous sorted-order C buffer (the TMA target)
    c_sorted = torch.zeros(
        (num_loras * EM, w1_output_dim_size),
        dtype=torch.bfloat16,
        device=device,
    )

    _clear_caches()

    kwargs = dict(
        c_sorted=c_sorted,
        a_intermediate_cache1=a_intermediate,
        lora_b_stacked=tensors["lora_b"],
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=ntpp,
        token_lora_mapping=tensors["token_lora_mapping"],
        top_k_num=top_k,
        lora_ids=tensors["lora_ids"],
        adapter_enabled=tensors["adapter_enabled"],
        device=device,
        N=lora_rank,
        M=M,
        EM=EM,
        K=tensors["qcurr"].shape[1],
        num_tokens=tensors["num_tokens"],
        num_experts=num_experts,
        num_slices=num_slices,
        max_lora_rank=lora_rank,
        w1_output_dim_size=w1_output_dim_size,
        block_size_m=expand_cfg["block_m"],
        block_size_n=expand_cfg["block_n"],
        block_size_k=expand_cfg["block_k"],
        group_size_m=expand_cfg["group_size_m"],
        num_warps=expand_cfg["num_warps"],
        num_stages=expand_cfg["num_stages"],
        num_active_loras=tensors["num_active_loras"],
        mul_routed_weight=mul_routed_weight,
    )

    try:
        return _timed(_expand_tma_c, kwargs, num_iters=num_iters)
    except Exception as e:
        print(f"    [ERR] tma_c {expand_cfg}: {e}")
        with contextlib.suppress(Exception):
            torch.accelerator.synchronize()
        return float("inf")


def _bench_baseline(
    tensors,
    expand_cfg,
    top_k,
    num_experts,
    max_loras,
    lora_rank,
    num_slices,
    block_size_m,
    num_iters,
    device,
    mul_routed_weight=False,
):
    """Benchmark the baseline pointer-C expand kernel. Returns us."""
    M = tensors["M"]
    naive = _should_use_naive(M, top_k, num_experts, max_loras)

    if naive:
        topk_weights = torch.rand(
            M, top_k, device=device, dtype=torch.float32
        )
        expert_ids = tensors["topk_ids"].view(-1)
        sorted_token_ids = None
        ntpp = None
        EM = tensors["num_tokens"] * block_size_m
        use_gdc = supports_pdl()
    else:
        topk_weights, sorted_token_ids, expert_ids, ntpp = _prepare_moe_data(
            tensors, block_size_m, top_k, num_experts, max_loras, device
        )
        EM = sorted_token_ids.shape[1]
        use_gdc = False

    use_tma = supports_tma(device)
    w1_output_dim_size = tensors["lora_b"][0].shape[2]

    if use_tma and num_slices > 1:
        set_triton_allocator(device)

    if use_tma and sorted_token_ids is not None:
        cache_shape = (num_slices, sorted_token_ids.shape[0], EM, lora_rank)
    else:
        cache_shape = (num_slices, M, top_k, lora_rank)
    a_intermediate = torch.randn(
        cache_shape, dtype=torch.bfloat16, device=device
    )

    expand_out = tensors["expand_out"].zero_()
    _clear_caches()

    kwargs = dict(
        output=expand_out,
        a_intermediate_cache1=a_intermediate,
        lora_b_stacked=tensors["lora_b"],
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=ntpp,
        token_lora_mapping=tensors["token_lora_mapping"],
        top_k_num=top_k,
        lora_ids=tensors["lora_ids"],
        adapter_enabled=tensors["adapter_enabled"],
        device=device,
        N=lora_rank,
        M=M,
        EM=EM,
        K=tensors["qcurr"].shape[1],
        num_tokens=tensors["num_tokens"],
        num_experts=num_experts,
        num_slices=num_slices,
        max_lora_rank=lora_rank,
        w1_output_dim_size=w1_output_dim_size,
        block_size_m=expand_cfg["block_m"],
        block_size_n=expand_cfg["block_n"],
        block_size_k=expand_cfg["block_k"],
        group_size_m=expand_cfg["group_size_m"],
        num_warps=expand_cfg["num_warps"],
        num_stages=expand_cfg["num_stages"],
        split_k=expand_cfg.get("split_k", 1),
        num_active_loras=tensors["num_active_loras"],
        mul_routed_weight=mul_routed_weight,
        offset=0,
        use_gdc=use_gdc,
        use_tma=use_tma,
    )

    try:
        return _timed(_fused_moe_lora_expand, kwargs, num_iters=num_iters)
    except Exception as e:
        print(f"    [ERR] baseline {expand_cfg}: {e}")
        with contextlib.suppress(Exception):
            torch.accelerator.synchronize()
        return float("inf")


# =====================================================================
# Tuning
# =====================================================================
def tune_w2_expand_tma_c(
    batch_sizes,
    hidden_size,
    intermediate_size,
    lora_rank,
    max_loras,
    top_k,
    num_experts,
    dtype,
    num_iters,
    active_loras,
    routing="uniform",
    num_expert_hit=None,
    compare=True,
):
    """Tune w2 expand TMA-C kernel and optionally compare with baseline."""
    shrink_K = intermediate_size
    expand_N = hidden_size
    num_slices = 1
    mul_routed_weight = True
    device = torch.device("cuda")

    expand_space = _get_search_space("expand", lora_rank, shrink_K, expand_N)
    print(f"  Expand search space: {len(expand_space)} configs")

    results = {}

    for bs in batch_sizes:
        naive = _should_use_naive(bs, top_k, num_experts, max_loras)
        if naive:
            print(f"\n  bs={bs}: SKIP (naive routing, TMA-C not applicable)")
            continue

        print(f"\n  batch_size={bs} (sorted routing)")

        tensors = _make_tensors(
            bs, shrink_K, expand_N, lora_rank, max_loras, num_experts,
            top_k, num_slices, dtype, device,
            active_loras=active_loras, routing=routing,
            mul_routed_weight=mul_routed_weight,
            num_expert_hit=num_expert_hit,
        )

        # ── Sweep TMA-C ──────────────────────────────────────────
        best_tma_time = float("inf")
        best_tma_cfg = None

        print("    Sweeping TMA-C...")
        for i, ec in enumerate(expand_space):
            _clear_caches()
            t = _bench_tma_c(
                tensors, ec, top_k, num_experts, max_loras,
                lora_rank, num_slices, ec["block_m"], num_iters, device,
                mul_routed_weight=mul_routed_weight,
            )
            if t < best_tma_time:
                best_tma_time = t
                best_tma_cfg = ec.copy()
            if (i + 1) % 50 == 0 or i == len(expand_space) - 1:
                print(
                    f"      TMA-C [{i + 1}/{len(expand_space)}] "
                    f"best={best_tma_time:.1f} us  "
                    f"M{best_tma_cfg['block_m']}_N{best_tma_cfg['block_n']}"
                    f"_K{best_tma_cfg['block_k']}_s{best_tma_cfg['num_stages']}"
                )
            if (i + 1) % 100 == 0:
                gc.collect()
                torch.accelerator.empty_cache()

        # ── Sweep baseline ────────────────────────────────────────
        best_base_time = float("inf")
        best_base_cfg = None

        if compare:
            print("    Sweeping baseline...")
            for i, ec in enumerate(expand_space):
                _clear_caches()
                t = _bench_baseline(
                    tensors, ec, top_k, num_experts, max_loras,
                    lora_rank, num_slices, ec["block_m"], num_iters, device,
                    mul_routed_weight=mul_routed_weight,
                )
                if t < best_base_time:
                    best_base_time = t
                    best_base_cfg = ec.copy()
                if (i + 1) % 50 == 0 or i == len(expand_space) - 1:
                    print(
                        f"      Base [{i + 1}/{len(expand_space)}] "
                        f"best={best_base_time:.1f} us  "
                        f"M{best_base_cfg['block_m']}"
                        f"_N{best_base_cfg['block_n']}"
                        f"_K{best_base_cfg['block_k']}"
                        f"_s{best_base_cfg['num_stages']}"
                    )
                if (i + 1) % 100 == 0:
                    gc.collect()
                    torch.accelerator.empty_cache()

        # ── Report ────────────────────────────────────────────────
        diff = best_base_time - best_tma_time if compare else 0
        pct = diff / best_base_time * 100 if compare and best_base_time > 0 else 0

        print(f"\n    TMA-C:    {best_tma_time:>8.1f} us  cfg={best_tma_cfg}")
        if compare:
            print(f"    Baseline: {best_base_time:>8.1f} us  cfg={best_base_cfg}")
            tag = "faster" if diff > 0 else "slower"
            print(f"    Delta:    {diff:+.1f} us ({pct:+.1f}% {tag})")

        results[bs] = {
            "tma_c_us": best_tma_time,
            "tma_c_cfg": best_tma_cfg,
            "baseline_us": best_base_time if compare else None,
            "baseline_cfg": best_base_cfg,
        }

        del tensors
        gc.collect()
        torch.accelerator.empty_cache()

    return results


# =====================================================================
# Main
# =====================================================================
def main(args):
    print("=" * 60)
    print("W2 Expand TMA-C vs Baseline Comparison")
    print("=" * 60)
    print(f"  GPU           : {torch.cuda.get_device_name()}")
    print(f"  hidden_size   : {args.hidden_size}")
    print(f"  intermediate  : {args.intermediate_size}")
    print(f"  lora_rank     : {args.lora_rank}")
    print(f"  max_loras     : {args.max_loras}")
    print(f"  active_loras  : {args.active_loras or args.max_loras}")
    print(f"  routing       : {args.routing}")
    print(f"  top_k         : {args.top_k}")
    print(f"  num_experts   : {args.num_experts}")
    print(f"  expert_hit    : {args.num_expert_hit or args.num_experts}")
    print(f"  batch_sizes   : {args.batch_sizes}")
    print(f"  num_iters     : {args.num_iters}")
    print(f"  compare       : {not args.tma_only}")
    print()

    start = time.time()

    results = tune_w2_expand_tma_c(
        batch_sizes=args.batch_sizes,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        lora_rank=args.lora_rank,
        max_loras=args.max_loras,
        top_k=args.top_k,
        num_experts=args.num_experts,
        dtype=torch.bfloat16,
        num_iters=args.num_iters,
        active_loras=args.active_loras,
        routing=args.routing,
        num_expert_hit=args.num_expert_hit,
        compare=not args.tma_only,
    )

    # ── Summary table ─────────────────────────────────────────────
    compare = not args.tma_only
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if compare:
        print(
            f"  {'bs':>6} | {'TMA-C (us)':>10} | {'Base (us)':>10} "
            f"| {'Delta':>10} | TMA-C config"
        )
        print("  " + "-" * 72)
    else:
        print(f"  {'bs':>6} | {'TMA-C (us)':>10} | TMA-C config")
        print("  " + "-" * 50)

    for bs in sorted(results.keys()):
        r = results[bs]
        tma = r["tma_c_us"]
        cfg = r["tma_c_cfg"]
        cs = (
            f"M{cfg['block_m']}_N{cfg['block_n']}"
            f"_K{cfg['block_k']}_s{cfg['num_stages']}"
            f"_g{cfg['group_size_m']}"
        )
        if compare and r["baseline_us"] is not None:
            base = r["baseline_us"]
            diff = base - tma
            pct = diff / base * 100 if base > 0 else 0
            print(
                f"  {bs:>6} | {tma:>10.1f} | {base:>10.1f} "
                f"| {pct:>+9.1f}% | {cs}"
            )
        else:
            print(f"  {bs:>6} | {tma:>10.1f} | {cs}")

    # Save TMA-C best configs
    tma_c_best = {bs: r["tma_c_cfg"] for bs, r in results.items() if r["tma_c_cfg"]}
    if tma_c_best:
        save_dir = getattr(args, "save_dir", "./tuned_w2_expand_tma_c_configs")
        _save_configs(tma_c_best, "w2_expand", args.max_loras, 1,
                      args.lora_rank, args.hidden_size, args.intermediate_size, save_dir)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")




def benchmark_w2_tma_c(args):
    """Load saved TMA-C configs and benchmark."""
    device = torch.device("cuda")
    config_dir = args.config_dir or "./tuned_w2_expand_tma_c_configs"
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")

    ef = os.path.join(config_dir, f"{gpu}_FUSED_MOE_LORA_W2_EXPAND.json")
    if not os.path.exists(ef):
        print(f"  [SKIP] TMA-C config not found: {ef}")
        return

    shrink_K = args.intermediate_size
    expand_N = args.hidden_size
    num_slices = 1

    print(f"  w2_expand TMA-C benchmark (slices={num_slices})")
    print(f"  {'bs':>6} | {'lat_us':>10} | {'routing':>7} | config")
    print("  " + "-" * 65)

    for bs in args.batch_sizes:
        if _should_use_naive(bs, args.top_k, args.num_experts, args.max_loras):
            print(f"  {bs:>6} | {'N/A':>10} | {'naive':>7} | (TMA-C requires sorted)")
            continue
        ec = _load_config(ef, args.max_loras, num_slices, args.lora_rank,
                          args.hidden_size, args.intermediate_size, bs)
        t = _make_tensors(bs, shrink_K, expand_N, args.lora_rank, args.max_loras,
                          args.num_experts, args.top_k, num_slices, torch.bfloat16,
                          device, active_loras=args.active_loras, routing=args.routing,
                          mul_routed_weight=True, num_expert_hit=args.num_expert_hit)
        lat = _bench_tma_c(t, ec, args.top_k, args.num_experts, args.max_loras,
                           args.lora_rank, num_slices, ec["block_m"],
                           args.num_iters, device, mul_routed_weight=True)
        es = (f"M{ec['block_m']}_N{ec['block_n']}"
              f"_K{ec['block_k']}_s{ec['num_stages']}"
              f"_g{ec['group_size_m']}")
        print(f"  {bs:>6} | {lat:>10.1f} | {'sorted':>7} | {es}")
        del t
        gc.collect()
        torch.accelerator.empty_cache()
    print()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Tune W2 expand TMA-C kernel and compare with baseline"
    )
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--max-loras", type=int, default=4)
    parser.add_argument("--active-loras", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=[64, 256, 512, 1024, 2048],
    )
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument(
        "--routing", type=str, default="uniform",
        choices=["uniform", "random"],
    )
    parser.add_argument("--num-expert-hit", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="./tuned_w2_expand_tma_c_configs")
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Benchmark mode: load saved TMA-C configs and measure",
    )
    parser.add_argument("--config-dir", type=str, default=None)
    parser.add_argument(
        "--tma-only", action="store_true",
        help="Only tune TMA-C (skip baseline comparison)",
    )
    args = parser.parse_args()
    if args.benchmark:
        benchmark_w2_tma_c(args)
    else:
        main(args)
