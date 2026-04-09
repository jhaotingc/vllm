# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tune and benchmark the W2 expand-only LoRA MoE Triton kernel.

Standalone testing script that sweeps expand kernel configurations
for the W2 (down-projection) expand path in isolation, without the
shrink kernel or coordinate descent. Useful for profiling and
optimizing the expand kernel independently.

Imports shared utilities from tune_lora_moe.py (must be in the same
directory or on PYTHONPATH).

Usage examples:

  # Tune w2 expand for Nemotron-3-Nano-like model
  python tune_lora_moe_w2_expand.py \\
      --hidden-size 3072 --intermediate-size 2944 \\
      --lora-rank 32 --max-loras 8 --top-k 4 --num-experts 32 \\
      --batch-sizes 64 256 512 1024 2048 \\
      --save-dir ./tuned_w2_expand_configs

  # Benchmark with saved configs
  python tune_lora_moe_w2_expand.py \\
      --hidden-size 3072 --intermediate-size 2944 \\
      --lora-rank 32 --max-loras 8 --top-k 4 --num-experts 32 \\
      --batch-sizes 64 256 512 1024 2048 \\
      --benchmark --config-dir ./tuned_w2_expand_configs
"""

import contextlib
import gc
import json
import os
import time

import torch

from vllm.lora.ops.triton_ops.fused_moe_lora_op import (
    _LORA_PTR_DICT,
    _fused_moe_lora_expand,
)
from vllm.lora.ops.triton_ops.utils import (
    _LORA_A_PTR_DICT,
    _LORA_B_PTR_DICT,
    supports_pdl,
    supports_tma,
)
from vllm.triton_utils.allocation import set_triton_allocator
from vllm.utils.argparse_utils import FlexibleArgumentParser

with contextlib.suppress(ImportError):
    import triton.tools.tensor_descriptor  # noqa: F401

# Import shared utilities from the main tuner
from tune_lora_moe import (
    _build_routing,
    _clear_caches,
    _get_search_space,
    _load_config,
    _make_tensors,
    _prepare_moe_data,
    _save_configs,
    _should_use_naive,
    _timed,
)


# ---------------------------------------------------------------------------
# Expand-only benchmark
# ---------------------------------------------------------------------------
def _bench_expand(
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
    """Benchmark the expand-only kernel. Returns us."""
    M = tensors["M"]
    naive = _should_use_naive(M, top_k, num_experts, max_loras)

    if naive:
        topk_weights = torch.rand(M, top_k, device=device, dtype=torch.float32)
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

    # Create intermediate cache (expand input)
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
        print(f"    [ERR] expand {expand_cfg}: {e}")
        with contextlib.suppress(Exception):
            torch.accelerator.synchronize()
        return float("inf")


# ---------------------------------------------------------------------------
# Expand-only tuning (exhaustive sweep)
# ---------------------------------------------------------------------------
def tune_w2_expand(
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
):
    """Tune w2 expand kernel configs via exhaustive sweep.

    W2 (down-projection) expand:
      - shrink_K = intermediate_size  (input to shrink / A dimension)
      - expand_N = hidden_size        (output dimension)
      - num_slices = 1                (always for w2)
      - mul_routed_weight = True      (w2 applies routing weights)
    """
    shrink_K = intermediate_size
    expand_N = hidden_size
    num_slices = 1
    mul_routed_weight = True
    device = torch.device("cuda")

    expand_space = _get_search_space("expand", lora_rank, shrink_K, expand_N)
    print(f"  Expand search space: {len(expand_space)} configs")

    expand_best_all = {}

    for bs in batch_sizes:
        print(f"\n  Tuning w2_expand | batch_size={bs}")

        tensors = _make_tensors(
            bs,
            shrink_K,
            expand_N,
            lora_rank,
            max_loras,
            num_experts,
            top_k,
            num_slices,
            dtype,
            device,
            active_loras=active_loras,
            routing=routing,
            mul_routed_weight=mul_routed_weight,
            num_expert_hit=num_expert_hit,
        )

        naive = _should_use_naive(bs, top_k, num_experts, max_loras)
        print(f"    routing={'naive' if naive else 'sorted'}")

        best_time = float("inf")
        best_cfg = None

        for i, ec in enumerate(expand_space):
            _clear_caches()
            t = _bench_expand(
                tensors,
                ec,
                top_k,
                num_experts,
                max_loras,
                lora_rank,
                num_slices,
                ec["block_m"],
                num_iters,
                device,
                mul_routed_weight=mul_routed_weight,
            )
            if t < best_time:
                best_time = t
                best_cfg = ec.copy()
            if (i + 1) % 50 == 0 or i == len(expand_space) - 1:
                print(
                    f"    [{i + 1}/{len(expand_space)}] "
                    f"best={best_time:.1f} us  cfg=M{best_cfg['block_m']}"
                    f"_N{best_cfg['block_n']}_K{best_cfg['block_k']}"
                    f"_s{best_cfg['num_stages']}_g{best_cfg['group_size_m']}"
                )
            if (i + 1) % 100 == 0:
                gc.collect()
                torch.accelerator.empty_cache()

        print(f"  => bs={bs}: {best_time:.1f} us")
        print(f"     expand={best_cfg}")
        expand_best_all[bs] = best_cfg

        del tensors
        gc.collect()
        torch.accelerator.empty_cache()

    return expand_best_all


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------
def benchmark_w2_expand(args):
    """Load tuned w2 expand configs and benchmark each batch size."""
    device = torch.device("cuda")
    config_dir = args.config_dir or args.save_dir
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")

    ef = os.path.join(
        config_dir, f"{gpu}_FUSED_MOE_LORA_W2_EXPAND.json"
    )
    if not os.path.exists(ef):
        print(f"  [SKIP] w2_expand config not found: {ef}")
        return

    shrink_K = args.intermediate_size
    expand_N = args.hidden_size
    num_slices = 1
    mul_routed_weight = True

    print(f"  w2_expand (slices={num_slices})")
    print(f"  {'bs':>6} | {'lat_us':>10} | {'routing':>7} | expand config")
    print("  " + "-" * 65)

    for bs in args.batch_sizes:
        ec = _load_config(
            ef,
            args.max_loras,
            num_slices,
            args.lora_rank,
            args.hidden_size,
            args.intermediate_size,
            bs,
        )
        t = _make_tensors(
            bs,
            shrink_K,
            expand_N,
            args.lora_rank,
            args.max_loras,
            args.num_experts,
            args.top_k,
            num_slices,
            torch.bfloat16,
            device,
            active_loras=args.active_loras,
            routing=args.routing,
            mul_routed_weight=mul_routed_weight,
            num_expert_hit=args.num_expert_hit,
        )
        naive = _should_use_naive(
            bs, args.top_k, args.num_experts, args.max_loras
        )
        lat = _bench_expand(
            t,
            ec,
            args.top_k,
            args.num_experts,
            args.max_loras,
            args.lora_rank,
            num_slices,
            ec["block_m"],
            args.num_iters,
            device,
            mul_routed_weight=mul_routed_weight,
        )
        route_tag = "naive" if naive else "sorted"
        es = (
            f"M{ec['block_m']}_N{ec['block_n']}"
            f"_K{ec['block_k']}_s{ec['num_stages']}"
            f"_g{ec['group_size_m']}"
        )
        print(f"  {bs:>6} | {lat:>10.1f} | {route_tag:>7} | {es}")
        del t
        gc.collect()
        torch.accelerator.empty_cache()
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    print("=" * 60)
    print("W2 Expand-Only LoRA MoE Kernel Tuning")
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
    print(f"  save_dir      : {args.save_dir}")
    print()

    start = time.time()

    expand_best = tune_w2_expand(
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
    )

    _save_configs(
        expand_best,
        "w2_expand",
        args.max_loras,
        1,  # num_slices (always 1 for w2)
        args.lora_rank,
        args.hidden_size,
        args.intermediate_size,
        args.save_dir,
    )

    elapsed = time.time() - start
    print(f"\nTotal tuning time: {elapsed:.1f}s")
    print(f"Configs saved to: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Tune W2 expand-only LoRA MoE Triton kernel configs"
    )
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument(
        "--intermediate-size",
        type=int,
        required=True,
        help="MoE intermediate size per partition",
    )
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--max-loras", type=int, default=4)
    parser.add_argument(
        "--active-loras",
        type=int,
        default=None,
        help="Active LoRAs (default: max-loras). "
        "Set to 1 for prefill simulation.",
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 16, 32, 64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument(
        "--routing",
        type=str,
        default="uniform",
        choices=["uniform", "random"],
    )
    parser.add_argument(
        "--num-expert-hit",
        type=int,
        default=None,
        help="Number of experts that receive tokens (default: all).",
    )
    parser.add_argument("--save-dir", type=str, default="./tuned_w2_expand_configs")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark mode: load configs and measure latency",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory to load tuned configs from (benchmark mode)",
    )
    args = parser.parse_args()
    if args.benchmark:
        benchmark_w2_expand(args)
    else:
        main(args)
