# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The static-shape SSD launch of a piecewise-captured Mamba2 mixer (whole
padded token buffer, padded sequences and chunks, initial states read from the
state cache through indices) must match the dynamic launch on the prefill
tokens exactly."""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cuda():
    pytest.skip("CUDA only", allow_module_level=True)

NHEADS, HEADDIM, NGROUPS, DSTATE, CHUNK_SIZE = 16, 64, 4, 128, 128
MAX_SEQS = 16


def _chunk_metadata(seqlens: list[int], computed: list[int]):
    """Prefill chunking of the Mamba2 builder: finish a partial chunk first."""
    cu, seq_idx, last = [0], [], []
    pos = 0
    for i, (n, c) in enumerate(zip(seqlens, computed)):
        if c % CHUNK_SIZE:
            take = min(-(-c // CHUNK_SIZE) * CHUNK_SIZE - c, n)
            pos += take
            cu.append(pos)
            seq_idx.append(i)
            n -= take
        while n > 0:
            take = min(CHUNK_SIZE, n)
            pos += take
            cu.append(pos)
            seq_idx.append(i)
            n -= take
        last.append(len(cu) - 2)
    return cu, seq_idx, last


@pytest.mark.parametrize(
    "num_decode_tokens,seqlens,computed,num_tokens",
    [
        (3, [1069], [0], 2048),
        (3, [1055], [300], 1152),
        (0, [512, 37, 300], [0, 64, 129], 1024),
        (9, [5, 1], [0, 7], 16),
        (6, [], [], 64),
        (0, [128, 128, 1], [128, 0, 1000], 512),
    ],
)
def test_static_ssd_matches_dynamic(num_decode_tokens, seqlens, computed, num_tokens):
    set_random_seed(0)
    device = torch.device("cuda")
    bf16, f16 = torch.bfloat16, torch.float16
    num_seqs = len(seqlens)
    start = num_decode_tokens
    end = num_decode_tokens + sum(seqlens)

    x = torch.randn(num_tokens, NHEADS, HEADDIM, dtype=bf16, device=device) * 0.5
    dt = torch.rand(num_tokens, NHEADS, dtype=bf16, device=device)
    A = -torch.rand(NHEADS, dtype=torch.float32, device=device)
    B = torch.randn(num_tokens, NGROUPS, DSTATE, dtype=bf16, device=device) * 0.5
    C = torch.randn(num_tokens, NGROUPS, DSTATE, dtype=bf16, device=device) * 0.5
    D = torch.rand(NHEADS, dtype=torch.float32, device=device)
    dt_bias = torch.rand(NHEADS, dtype=torch.float32, device=device) - 2
    # State cache with padded page rows, like vLLM's Mamba pages.
    num_slots = 2 * MAX_SEQS + 4
    row = NHEADS * HEADDIM * DSTATE
    page = torch.randn(num_slots, row + 256, dtype=f16, device=device) * 0.1
    page_ref = page.clone()
    cache = page[:, :row].view(num_slots, NHEADS, HEADDIM, DSTATE)
    cache_ref = page_ref[:, :row].view(num_slots, NHEADS, HEADDIM, DSTATE)
    slots = (torch.randperm(num_slots - 1, device=device)[:num_seqs] + 1).int()
    has_init = torch.tensor([c > 0 for c in computed], dtype=torch.bool, device=device)

    cu, seq_idx, last = _chunk_metadata(seqlens, computed)
    out_ref = torch.full((num_tokens, NHEADS, HEADDIM), 7.0, dtype=bf16, device=device)
    if num_seqs:
        as_i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
        mamba_chunk_scan_combined_varlen(
            x[start:end],
            dt[start:end],
            A,
            B[start:end],
            C[start:end],
            chunk_size=CHUNK_SIZE,
            cu_seqlens=as_i32([0, *torch.tensor(seqlens).cumsum(0).tolist()]),
            cu_chunk_seqlens=as_i32(cu),
            last_chunk_indices=as_i32(last),
            seq_idx=as_i32(seq_idx),
            out=out_ref[start:end],
            D=D,
            dt_bias=dt_bias,
            initial_states=torch.where(
                has_init[:, None, None, None], cache_ref[slots], 0
            )
            if any(computed)
            else None,
            dt_softplus=True,
            state_dtype=f16,
            final_states_out=cache_ref,
            final_state_indices=slots,
        )

    # Static launch: MAX_SEQS sequence slots and the static chunk bound.
    seq_slots = min(MAX_SEQS, num_tokens)
    num_chunks = -(-num_tokens // CHUNK_SIZE) + 2 * seq_slots
    pad_chunks = num_chunks - len(seq_idx)
    as_i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    cu_seqlens = [start + c for c in [0, *torch.tensor(seqlens).cumsum(0).tolist()]]
    static_slots = torch.full((seq_slots,), -1, dtype=torch.int32, device=device)
    static_slots[:num_seqs] = slots
    static_has_init = torch.zeros(seq_slots, dtype=torch.int32, device=device)
    static_has_init[:num_seqs] = has_init.int()
    out = torch.full_like(out_ref, 7.0)
    mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=as_i32(cu_seqlens + [end] * (seq_slots - num_seqs)),
        cu_chunk_seqlens=as_i32([start + c for c in cu] + [-1] * pad_chunks),
        last_chunk_indices=as_i32(last + [len(seq_idx) - 1] * (seq_slots - num_seqs)),
        seq_idx=as_i32(seq_idx + [-1] * pad_chunks),
        out=out,
        D=D,
        dt_bias=dt_bias,
        initial_states=cache,
        initial_state_indices=static_slots,
        has_initial_states=static_has_init,
        dt_softplus=True,
        state_dtype=f16,
        final_states_out=cache,
        final_state_indices=static_slots,
        has_pad_chunks=True,
    )

    # Same chunks and kernels: bitwise equal. Padding, decode rows, other
    # cache slots and page padding are untouched.
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(page, page_ref, rtol=0, atol=0)
