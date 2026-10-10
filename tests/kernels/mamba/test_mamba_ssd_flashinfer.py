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
    mamba_chunk_scan_flashinfer,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

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
