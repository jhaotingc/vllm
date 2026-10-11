# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer's fused SSD prefill scan (--mamba-ssd-backend flashinfer) against
vLLM's Triton varlen scan, on MambaMixer2-shaped inputs."""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.mamba.ops.ssd_flashinfer import (
    FLASHINFER_SSD_HEAD_DIM,
    FLASHINFER_SSD_STATE_SIZE,
    allocate_static_ssd_scratch,
    mamba_chunk_scan_flashinfer,
    mamba_chunk_scan_flashinfer_static,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mamba2_attn import mamba2_static_flashinfer_seqs

if not current_platform.is_device_capability_family(100):
    pytest.skip("FlashInfer's fused SSD needs SM100/SM103", allow_module_level=True)
pytest.importorskip("flashinfer.mamba.ssd_combined")

CHUNK_SIZE = 128


def _chunk_metadata(seqlens: list[int], device: torch.device):
    cu_chunk, seq_idx, last = [0], [], []
    for i, n in enumerate(seqlens):
        start = cu_chunk[-1]
        for offset in range(0, n, CHUNK_SIZE):
            cu_chunk.append(start + min(n, offset + CHUNK_SIZE))
            seq_idx.append(i)
        last.append(len(cu_chunk) - 2)
    as_i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    return as_i32(cu_chunk), as_i32(seq_idx), as_i32(last)


@pytest.mark.parametrize(
    "seqlens", [[1069], [5], [1069, 37, 300], [128, 128, 1]], ids=str
)
@pytest.mark.parametrize("with_initial_states", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
def test_flashinfer_ssd_matches_triton(seqlens, with_initial_states, state_dtype):
    set_random_seed(0)
    device = torch.device("cuda")
    nheads, ngroups = 16, 4
    headdim, dstate = FLASHINFER_SSD_HEAD_DIM, FLASHINFER_SSD_STATE_SIZE
    bf16 = torch.bfloat16
    num_tokens = sum(seqlens)
    batch = len(seqlens)

    # x/B/C are strided slices of one conv1d output row, dt of in_proj's.
    conv_dim = nheads * headdim + 2 * ngroups * dstate
    xBC = torch.randn(num_tokens, conv_dim, dtype=bf16, device=device) * 0.5
    x = xBC[:, : nheads * headdim].view(num_tokens, nheads, headdim)
    B = xBC[:, nheads * headdim : -ngroups * dstate].view(num_tokens, ngroups, dstate)
    C = xBC[:, -ngroups * dstate :].view(num_tokens, ngroups, dstate)
    dt = torch.randn(num_tokens, 2 * nheads, dtype=bf16, device=device)[:, nheads:]
    A = -torch.rand(nheads, dtype=torch.float32, device=device) * 2
    D = torch.rand(nheads, dtype=torch.float32, device=device)
    dt_bias = torch.rand(nheads, dtype=torch.float32, device=device) - 2

    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seqlens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cu_chunk, seq_idx, last_chunk = _chunk_metadata(seqlens, device)

    # Sequences live in scattered cache slots; odd ones carry a prior state.
    num_slots = 2 * batch + 3
    slots = torch.randperm(num_slots, device=device)[:batch].to(torch.int32)
    cache = torch.randn(
        num_slots, nheads, headdim, dstate, dtype=state_dtype, device=device
    )
    cache *= 0.1
    has_initial = torch.arange(batch, device=device) % 2 == 1
    if not with_initial_states:
        has_initial.zero_()

    ref_cache = cache.clone()
    initial_states = (
        torch.where(has_initial[:, None, None, None], ref_cache[slots], 0)
        if with_initial_states
        else None
    )
    ref_out = torch.empty(num_tokens, nheads, headdim, dtype=bf16, device=device)
    mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunk,
        last_chunk_indices=last_chunk,
        seq_idx=seq_idx,
        out=ref_out,
        D=D,
        dt_bias=dt_bias,
        initial_states=initial_states,
        dt_softplus=True,
        state_dtype=state_dtype,
        final_states_out=ref_cache,
        final_state_indices=slots,
    )

    out = torch.empty_like(ref_out)
    mamba_chunk_scan_flashinfer(
        x,
        dt,
        A,
        B,
        C,
        D.to(bf16),
        dt_bias,
        cu_seqlens,
        out,
        cache,
        slots,
        has_initial if with_initial_states else None,
    )

    # Different chunking and accumulation order; D is bfloat16 here.
    torch.testing.assert_close(out, ref_out, atol=0.13, rtol=0.05)
    torch.testing.assert_close(cache, ref_cache, atol=0.02, rtol=0.02)
    untouched = torch.ones(num_slots, dtype=torch.bool, device=device)
    untouched[slots.long()] = False
    assert torch.equal(cache[untouched], ref_cache[untouched])


@pytest.mark.parametrize(
    "num_decode_tokens,seqlens,computed,num_tokens",
    [
        (3, [1069], [0], 2048),
        (12, [500, 37, 300], [0, 64, 0], 1024),
        (0, [128, 128, 1], [128, 0, 1000], 512),
        (9, [], [], 64),
    ],
)
def test_flashinfer_ssd_static_graph_matches_eager(
    num_decode_tokens, seqlens, computed, num_tokens
):
    """The static FlashInfer scan of a piecewise-captured mixer, over the whole
    padded batch [decode tokens | prefill tokens | padding] and captured in a
    CUDA graph, matches the eager FlashInfer scan of the prefill tokens on
    every replay, also after the step's layout and inputs change. The decode
    and padding rows are identity rows of the first and the last sequence."""
    set_random_seed(0)
    device = torch.device("cuda")
    nheads, ngroups = 16, 4
    headdim, dstate = FLASHINFER_SSD_HEAD_DIM, FLASHINFER_SSD_STATE_SIZE
    bf16, f16, i32 = torch.bfloat16, torch.float16, torch.int32
    static_seqs = 6
    num_cache_slots = 2 * static_seqs + 3
    batch = len(seqlens)
    conv_dim = nheads * headdim + 2 * ngroups * dstate

    xBC = torch.empty(num_tokens, conv_dim, dtype=bf16, device=device)
    x = xBC[:, : nheads * headdim].view(num_tokens, nheads, headdim)
    B = xBC[:, nheads * headdim : -ngroups * dstate].view(num_tokens, ngroups, dstate)
    C = xBC[:, -ngroups * dstate :].view(num_tokens, ngroups, dstate)
    dt_full = torch.empty(num_tokens, 2 * nheads, dtype=bf16, device=device)
    dt = dt_full[:, nheads:]
    A = -torch.rand(nheads, dtype=torch.float32, device=device) * 2
    D = torch.rand(nheads, dtype=torch.float32, device=device).to(bf16)
    dt_bias = torch.rand(nheads, dtype=torch.float32, device=device) - 2
    cache = torch.empty(
        num_cache_slots, nheads, headdim, dstate, dtype=f16, device=device
    )
    out = torch.empty(num_tokens, nheads, headdim, dtype=bf16, device=device)
    # Persistent metadata the graph reads.
    cu_seqlens_fi = torch.empty(static_seqs + 1, dtype=i32, device=device)
    prefill_bounds = torch.empty(2, dtype=i32, device=device)
    state_indices = torch.empty(static_seqs, dtype=i32, device=device)
    has_initial = torch.empty(static_seqs, dtype=i32, device=device)
    num_seqs = torch.empty(1, dtype=i32, device=device)
    allocate_static_ssd_scratch(device, static_seqs, num_tokens, nheads, f16)

    def stage(start):
        """New inputs and a layout with the prefills from token `start`."""
        xBC.copy_(torch.randn_like(xBC) * 0.5)
        dt_full.copy_(torch.randn_like(dt_full))
        cache.copy_(torch.randn_like(cache) * 0.1)
        slots = torch.randperm(num_cache_slots, device=device)[:batch].to(i32)
        offsets = [start, *(start + torch.tensor(seqlens).cumsum(0)).tolist()]
        end = offsets[-1]
        # The mixer's conv1d leaves the rows outside the prefill sequences
        # unwritten, so they can hold any bits, NaN included.
        xBC[:start] = float("nan")
        xBC[end:] = float("nan")
        dt_full[end:] = float("nan")
        cu_seqlens_fi.copy_(
            torch.tensor(
                [0, *offsets[1:-1], *[num_tokens] * (static_seqs - max(batch, 1) + 1)],
                dtype=i32,
            )
        )
        prefill_bounds.copy_(torch.tensor([start, end], dtype=i32))
        state_indices.fill_(-1)
        state_indices[:batch] = slots
        has_initial.zero_()
        has_initial[:batch] = torch.tensor([c > 0 for c in computed], dtype=i32)
        num_seqs.fill_(batch)
        return slots, offsets

    def reference(slots, offsets):
        ref_cache = cache.clone()
        ref_out = None
        if batch:
            start, end = offsets[0], offsets[-1]
            ref_out = torch.empty(
                end - start, nheads, headdim, dtype=bf16, device=device
            )
            has = has_initial[:batch].bool()
            mamba_chunk_scan_flashinfer(
                x[start:end],
                dt[start:end],
                A,
                B[start:end],
                C[start:end],
                D,
                dt_bias,
                torch.tensor([o - start for o in offsets], dtype=i32, device=device),
                ref_out,
                ref_cache,
                slots,
                has if bool(has.any()) else None,
            )
        return ref_cache, ref_out

    def run():
        mamba_chunk_scan_flashinfer_static(
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            cu_seqlens_fi,
            prefill_bounds,
            out,
            cache,
            state_indices,
            has_initial,
            num_seqs,
        )

    stream = torch.cuda.Stream()
    stage(num_decode_tokens)
    with torch.cuda.stream(stream):
        run()  # warmup outside the capture, as the mixer's SSD warmup does
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()

    for start in (num_decode_tokens, num_decode_tokens + 1):
        if start + sum(seqlens) > num_tokens:
            continue
        slots, offsets = stage(start)
        ref_cache, ref_out = reference(slots, offsets)
        graph.replay()
        torch.accelerator.synchronize()
        if batch:
            # The batch layout changes FlashInfer's segmentation: bf16
            # rounding of a few outputs differs.
            torch.testing.assert_close(
                out[offsets[0] : offsets[-1]], ref_out, atol=6e-2, rtol=5e-2
            )
        torch.testing.assert_close(cache, ref_cache, atol=2e-2, rtol=2e-2)
        untouched = torch.ones(num_cache_slots, dtype=torch.bool, device=device)
        untouched[slots.long()] = False
        assert torch.equal(cache[untouched], ref_cache[untouched])


def test_static_flashinfer_seqs_follow_the_size():
    """The sequence slots of the static FlashInfer scan never exceed the
    static launch's and never shrink as the size grows (the model runner
    bounds a step's slots by those of its unpadded size)."""
    for max_num_seqs in (8, 128, 256):
        previous = 0
        for num_tokens in range(1, 40000, 37):
            slots = mamba2_static_flashinfer_seqs(num_tokens, max_num_seqs)
            assert 1 <= slots <= min(max_num_seqs, num_tokens)
            assert slots >= previous
            previous = slots
