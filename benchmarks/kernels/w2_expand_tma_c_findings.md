# W2 Expand TMA-C Kernel: Findings

**Date:** 2026-04-08
**GPU:** NVIDIA H200 (SM90, 143 GB HBM3e)
**Model config:** hidden=3072, intermediate=2944, lora_rank=32, max_loras=8, top_k=4, num_experts=32
**Routing:** uniform

## Background

The fused MoE LoRA expand kernel (`_fused_moe_lora_kernel` with `ADD_INPUTS=True`)
accumulates LoRA output into the MoE output tensor. The W2 (down-projection)
expand path does:

```python
prev = tl.load(c_ptrs, mask=c_mask, other=0.0)
tl.store(c_ptrs, prev + accumulator, mask=c_mask)
```

where `c_ptrs` are computed from scattered `sorted_token_ids` indices via
`_get_c_ptrs()`. This pointer arithmetic consumes registers for per-element
address computation.

**Hypothesis:** Replacing pointer-based C access with TMA descriptor-based
load+store reduces register pressure and improves performance at larger batch
sizes, where occupancy is the bottleneck.

## Experiment Design

Created `_expand_tma_c_kernel` — a variant where:
- Output C is a contiguous 2D buffer in sorted token order: `(max_loras * EM, N)`
- C uses `TensorDescriptor.from_tensor()` for TMA load+add+store
- A and B paths remain **identical** to the baseline (TMA for both)
- The kernel writes to sorted order; an unpermute would be needed in production

Both TMA-C and baseline were independently tuned across 216 configs per batch size.

---

## Case 1: Decode (active_loras=8)

8 LoRA adapters active, tokens spread across all adapters (multi-tenant serving).

| bs | Baseline (us) | Baseline regs | Baseline config | TMA-C (us) | TMA-C regs | TMA-C config | Speedup |
|---|---|---|---|---|---|---|---|
| 64 | 24.0 | 94 | M16_N256_K32_s2_g64 | 31.4 | 86 | M16_N256_K32_s2_g64 | **-30.7%** |
| 256 | 27.9 | 72 | M16_N256_K16_s2_g64 | 32.3 | 64 | M16_N256_K16_s3_g64 | **-15.6%** |
| 512 | 31.4 | 72 | M16_N256_K16_s2_g64 | 34.0 | 64 | M16_N256_K16_s3_g64 | **-8.2%** |
| 1024 | 41.5 | 72 | M16_N256_K16_s3_g64 | 38.0 | 64 | M16_N256_K16_s3_g1 | **+8.3%** |
| 2048 | 69.5 | 128 | M32_N256_K16_s3_g1 | 63.2 | 124 | M32_N256_K32_s2_g64 | **+9.0%** |

### Decode observations

- **Register reduction:** 64 vs 72 at bs=256-1024 (-11%), 124 vs 128 at bs=2048 (-3%)
- **Occupancy gain:** 64 regs → 8 blocks/SM vs 72 regs → 7 blocks/SM (+14%) at bs=256-1024
- **Crossover at ~bs=768:** TMA descriptor creation overhead dominates below this
- At bs=2048, TMA-C uses K=32 (full lora_rank, eliminates K-loop) vs baseline K=16

---

## Case 2: Prefill (active_loras=1)

All tokens use the same LoRA adapter (single-request prefill).

| bs | Baseline (us) | Baseline regs | Baseline config | TMA-C (us) | TMA-C regs | TMA-C config | Speedup |
|---|---|---|---|---|---|---|---|
| 64 | 8.7 | 72 | M16_N256_K16_s2_g16 | 9.3 | 86 | M16_N256_K32_s2_g64 | **-6.0%** |
| 256 | 14.2 | 136 | M32_N256_K32_s2_g32 | 12.7 | 124 | M32_N256_K32_s2_g64 | **+10.6%** |
| 512 | 23.2 | 136 | M32_N256_K32_s2_g1 | 19.9 | 124 | M32_N256_K32_s2_g8 | **+14.4%** |
| 1024 | 42.0 | 136 | M32_N256_K32_s2_g64 | 34.5 | 124 | M32_N256_K32_s2_g64 | **+18.0%** |
| 2048 | 89.5 | 128 | M32_N256_K16_s3_g1 | 82.9 | 124 | M32_N256_K32_s2_g64 | **+7.4%** |

### Prefill observations

- **Register reduction:** 124 vs 136 at bs=256-1024 (-9%), crossing the 4→5 blocks/SM boundary
- **Crossover at ~bs=128:** Much lower than decode because single-lora packing is denser
- **Peak speedup +18.0% at bs=1024:** The sweet spot where occupancy gain meets sufficient grid parallelism
- **M=32 dominates:** Unlike decode (M=16), prefill has enough tokens per lora for larger tiles
- **K=32 everywhere for TMA-C:** Full lora_rank tile, zero K-loop overhead

---

## Decode vs Prefill comparison

| bs | Decode speedup | Prefill speedup |
|---|---|---|
| 64 | -30.7% | -6.0% |
| 256 | -15.6% | **+10.6%** |
| 512 | -8.2% | **+14.4%** |
| 1024 | +8.3% | **+18.0%** |
| 2048 | +9.0% | +7.4% |

Prefill benefits more because:
1. All tokens in one lora bucket → denser, more TMA-friendly access
2. Lower crossover point (bs=128 vs bs=768)
3. Larger register reduction at M=32 configs (124 vs 136 = -9%)

---

## Production dispatch recommendation

| Scenario | Threshold | Kernel |
|---|---|---|
| W2 prefill (1 active LoRA) | bs >= 128 | TMA-C |
| W2 prefill (1 active LoRA) | bs < 128 | Baseline |
| W2 decode (8 active LoRAs) | bs >= 768 | TMA-C |
| W2 decode (8 active LoRAs) | bs < 768 | Baseline |

## Production considerations

1. **Sorted-order C buffer required** — `(max_loras * EM, hidden_size)` contiguous allocation
2. **Unpermute kernel needed** — scatter results back to original token order after expand
3. **Unpermute cost** — Python-loop permute is ~1200 us (prohibitive); needs fused CUDA gather/scatter

## Files

- `tune_lora_moe_w2_expand_tma_c.py` — TMA-C kernel + comparison benchmark
- `tune_lora_moe_w2_expand.py` — Baseline expand-only benchmark
- `ncu_profile_w2_expand.py` — ncu profiling helper for register counts
- `tune_lora_moe.py` — Shared utilities (from vLLM upstream)


---

## Occupancy analysis (SM90, 65536 regs/SM, 128 threads/block)

max_blocks = floor(65536 / (regs_per_thread * threads_per_block))

### Decode (active_loras=8)

| bs | Baseline regs | Baseline max blocks | TMA-C regs | TMA-C max blocks | Occ. gain |
|---|---|---|---|---|---|
| 64 | 94 | **5** | 86 | **5** | 0% |
| 256 | 72 | **7** | 64 | **8** | **+14%** |
| 512 | 72 | **7** | 64 | **8** | **+14%** |
| 1024 | 72 | **7** | 64 | **8** | **+14%** |
| 2048 | 128 | **4** | 124 | **4** | 0% |

At bs=256-1024, TMA-C crosses from 7 to 8 blocks/SM. This +14% occupancy gain
materializes as +8.3% speedup at bs=1024 where grid parallelism is sufficient.

At bs=64 and bs=2048, both kernels land in the same occupancy tier despite
different register counts, so the occupancy gain is zero.

### Prefill (active_loras=1)

| bs | Baseline regs | Baseline max blocks | TMA-C regs | TMA-C max blocks | Occ. gain |
|---|---|---|---|---|---|
| 64 | 72 | **7** | 86 | **5** | **-29%** (worse) |
| 256 | 136 | **3** | 124 | **4** | **+33%** |
| 512 | 136 | **3** | 124 | **4** | **+33%** |
| 1024 | 136 | **3** | 124 | **4** | **+33%** |
| 2048 | 128 | **4** | 124 | **4** | 0% |

At bs=256-1024 prefill, baseline uses 136 regs (M32_K32 config) which limits it
to only 3 blocks/SM. TMA-C uses 124 regs for the same tile shape, crossing the
boundary to 4 blocks/SM -- a +33% occupancy gain. This is why prefill sees
up to +18% speedup vs decode +8.3%: the occupancy boundary crossing is
more impactful (3->4 vs 7->8).

At bs=64 prefill, TMA-C actually has worse occupancy (86 regs = 5 blocks vs
baseline 72 regs = 7 blocks) because TMA-C selects K=32 while baseline uses
K=16. The smaller grid at bs=64 cannot exploit the K=32 benefit.


---

## Why TMA-C reduces registers

### Baseline pointer-C path (what the current kernel does)

In the baseline expand kernel, storing the output tile requires computing
a unique memory address for every element in the BLOCK_M x BLOCK_N tile.
The C pointer computation in `_get_c_ptrs` is:

```
c_ptrs = cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
```

where `offs_token` is a BLOCK_M-element vector loaded from `sorted_token_ids`
(scattered, non-contiguous indices). This expands to a BLOCK_M x BLOCK_N
pointer matrix. The compiler must keep these address registers live from the
point they are computed through the final `tl.store`, which includes:

1. **offs_token** (BLOCK_M i64 values) -- loaded from sorted_token_ids
2. **offs_cn** (BLOCK_N i32 values) -- pid_n * BLOCK_SIZE_N + arange
3. **c_ptrs** (BLOCK_M x BLOCK_N i64 pointer matrix) -- the outer product
4. **c_mask** (BLOCK_M x BLOCK_N i1 mask) -- token_mask[:, None] & (offs_cn < N)
5. **prev** (BLOCK_M x BLOCK_N bf16 values) -- loaded for ADD_INPUTS

For a M16_N256 tile, that is:
- 16 * 256 = 4096 pointer values (i64, 2 regs each = 8192 register slots)
- Plus the mask, prev load, and intermediate address arithmetic

The compiler applies register allocation optimizations (spilling, reuse), but
the fundamental pressure from maintaining a 2D scattered address matrix is
significant.

### TMA-C path (what the new kernel does)

With TMA, the output tile access replaces all of the above with:

```
offs_cm = lora_id * EM + pid_m * BLOCK_SIZE_M   # 1 scalar
offs_cn = pid_n * BLOCK_SIZE_N                   # 1 scalar
prev = c_desc.load([offs_cm, offs_cn])
c_desc.store([offs_cm, offs_cn], prev + accumulator)
```

The TMA hardware unit on SM90 takes 2 scalar offsets and handles:
- 2D address calculation (base + row_offset * stride + col_offset)
- Bounds checking (replaces the software mask)
- Data movement between global memory and registers

This replaces ~BLOCK_M * BLOCK_N address registers with 2 scalar offset
registers. The `prev` tile still occupies BLOCK_M * BLOCK_N registers for
the loaded data, but the address computation overhead is eliminated.

### Where the savings come from (register breakdown)

| Component | Baseline (pointer-C) | TMA-C | Saved |
|---|---|---|---|
| C address matrix (c_ptrs) | BLOCK_M * BLOCK_N * 2 regs (i64) | 2 scalar regs | ~most of the diff |
| C mask (c_mask) | BLOCK_M * BLOCK_N / 32 regs (i1 packed) | 0 (TMA handles bounds) | small |
| offs_token for C indexing | BLOCK_M regs (i64) | 0 (not used for C) | small |
| offs_cn for C indexing | BLOCK_N regs (i32) | 0 (not used for C) | small |
| prev data tile | BLOCK_M * BLOCK_N regs | same (still loaded) | 0 |
| C tile offsets | 0 | 2 scalar regs | -2 (negligible) |

Note: offs_token is still needed in TMA-C for routing weight lookup
(MUL_ROUTED_WEIGHT), so it is not fully eliminated -- only its use in
C address computation is removed. The compiler may still keep it live
but can more aggressively schedule its lifetime.

### Why the measured reduction is modest (8-12 regs, not hundreds)

The theoretical address matrix is BLOCK_M * BLOCK_N * 2 = 8192 register
slots for M16_N256. In practice, the compiler heavily optimizes this:

1. **Register reuse:** The compiler does not materialize all 4096 pointers
   simultaneously. It computes c_ptrs in chunks, stores each chunk, then
   reuses those registers.
2. **Spilling:** Some address values are spilled to local memory and reloaded.
3. **Instruction scheduling:** The compiler interleaves address computation
   with the GEMM loop to overlap register lifetimes.

So the actual register overhead of pointer-C is not 8192 but rather the
peak live register count during the store phase, which is typically
20-40 registers above what TMA-C needs. The ncu measurements (8-12 reg
reduction) reflect this optimized reality.

The modest per-register reduction still crosses occupancy boundaries
(7->8 blocks in decode, 3->4 in prefill) which yields measurable speedups.


---

## Reproducing the results

### Prerequisites

- NVIDIA H200 GPU (SM90)
- vLLM installed from source at `/scratch/vllm_workspace_dev/vllm-dev`
- Working directory: `benchmarks/kernels/`

### Step 1: Tune + compare (decode, active_loras=8)

```bash
cd /scratch/vllm_workspace_dev/vllm-dev/benchmarks/kernels

python3 tune_lora_moe_w2_expand_tma_c.py \
    --hidden-size 3072 --intermediate-size 2944 \
    --lora-rank 32 --max-loras 8 --top-k 4 --num-experts 32 \
    --batch-sizes 64 256 512 1024 2048 --num-iters 50
```

### Step 2: Tune + compare (prefill, active_loras=1)

```bash
python3 tune_lora_moe_w2_expand_tma_c.py \
    --hidden-size 3072 --intermediate-size 2944 \
    --lora-rank 32 --max-loras 8 --top-k 4 --num-experts 32 \
    --batch-sizes 64 256 512 1024 2048 --num-iters 50 \
    --active-loras 1
```

### Step 3: Collect register counts with ncu

For decode (active_loras=8, default):

```bash
# Example for bs=1024, TMA-C best config
ncu --profile-from-start off \
    --metrics launch__registers_per_thread \
    -k regex:expand_tma_c \
    python3 ncu_profile_w2_expand.py \
        --kernel tma_c --batch-size 1024 \
        --config { block_m:16,block_n:256,block_k:16,group_size_m:1,num_warps:4,num_stages:3}

# Example for bs=1024, baseline best config
ncu --profile-from-start off \
    --metrics launch__registers_per_thread \
    -k regex:fused_moe_lora \
    python3 ncu_profile_w2_expand.py \
        --kernel baseline --batch-size 1024 \
        --config { block_m:16,block_n:256,block_k:16,group_size_m:64,num_warps:4,num_stages:3}
```

For prefill (active_loras=1):

```bash
# Example for bs=1024, TMA-C best config
ncu --profile-from-start off \
    --metrics launch__registers_per_thread \
    -k regex:expand_tma_c \
    python3 ncu_profile_w2_expand.py \
        --kernel tma_c --batch-size 1024 --active-loras 1 \
        --config { block_m:32,block_n:256,block_k:32,group_size_m:64,num_warps:4,num_stages:2}

# Example for bs=1024, baseline best config
ncu --profile-from-start off \
    --metrics launch__registers_per_thread \
    -k regex:fused_moe_lora \
    python3 ncu_profile_w2_expand.py \
        --kernel baseline --batch-size 1024 --active-loras 1 \
        --config { block_m:32,block_n:256,block_k:32,group_size_m:64,num_warps:4,num_stages:2}
```

### Step 4: Baseline-only tuning (optional)

```bash
python3 tune_lora_moe_w2_expand.py \
    --hidden-size 3072 --intermediate-size 2944 \
    --lora-rank 32 --max-loras 8 --top-k 4 --num-experts 32 \
    --batch-sizes 64 256 512 1024 2048
```

### TMA-C only (skip baseline comparison, faster)

```bash
python3 tune_lora_moe_w2_expand_tma_c.py \
    --hidden-size 3072 --intermediate-size 2944 \
    --lora-rank 32 --max-loras 8 --top-k 4 --num-experts 32 \
    --batch-sizes 64 256 512 1024 2048 --tma-only
```


---

## Case 3: Prefill with sparse expert routing (active_loras=1, expert_hit=16/32)

Real MoE routing is sparse -- not all 32 experts receive tokens. With
`num_expert_hit=16`, only half the experts get any tokens, producing
sparser M-tiles and fewer total thread blocks per lora bucket.

| bs | Baseline (us) | Baseline config | TMA-C (us) | TMA-C config | Speedup |
|---|---|---|---|---|---|
| 64 | 8.1 | M16_N256_K32_s2_g64 | 8.0 | M16_N256_K32_s2_g1 | **+1.2%** |
| 256 | 14.2 | M32_N256_K32_s2_g32 | 12.7 | M32_N256_K32_s2_g64 | **+10.8%** |
| 512 | 23.1 | M32_N256_K32_s2_g1 | 19.8 | M32_N256_K32_s2_g4 | **+14.3%** |
| 1024 | 41.8 | M32_N256_K32_s2_g4 | 34.3 | M32_N256_K32_s2_g1 | **+17.9%** |
| 2048 | 89.1 | M32_N256_K16_s3_g1 | 82.2 | M32_N256_K32_s2_g8 | **+7.8%** |

### Comparison: 32 expert hit vs 16 expert hit (prefill, active_loras=1)

| bs | 32 experts TMA-C speedup | 16 experts TMA-C speedup | Delta |
|---|---|---|---|
| 64 | -6.0% | **+1.2%** | +7.2pp |
| 256 | +10.6% | +10.8% | +0.2pp |
| 512 | +14.4% | +14.3% | -0.1pp |
| 1024 | +18.0% | +17.9% | -0.1pp |
| 2048 | +7.4% | +7.8% | +0.4pp |

### Observations

1. **Near-identical speedups at bs>=256**: Sparse expert routing does not
   meaningfully change TMA-C's advantage. The occupancy boundary crossing
   (136 to 124 regs = 3 to 4 blocks/SM) is the same regardless of expert sparsity.

2. **bs=64 flips from -6.0% to +1.2%**: With fewer experts hit, the grid is
   smaller and both kernels run faster (8.0 us vs 9.3 us for TMA-C). The
   reduced grid makes TMA descriptor overhead proportionally smaller,
   allowing TMA-C to break even.

3. **Absolute latencies are nearly identical**: Sparse routing produces the
   same amount of total work (same number of tokens * top_k GEMM tiles),
   just distributed across fewer experts. The kernel performance is driven
   by total tile count, not expert distribution.

### Repro: Prefill with sparse expert routing

```bash
python3 tune_lora_moe_w2_expand_tma_c.py \
    --hidden-size 3072 --intermediate-size 2944 \
    --lora-rank 32 --max-loras 8 --top-k 4 --num-experts 32 \
    --batch-sizes 64 256 512 1024 2048 --num-iters 50 \
    --active-loras 1 --num-expert-hit 16
```
