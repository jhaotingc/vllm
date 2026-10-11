# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 prefill SSD scan through FlashInfer's fused SSD (Cake backend).

Runs the chunked scan of ``mamba_chunk_scan_combined_varlen`` in two kernel
launches instead of five, with the packed-varlen ``cu_seqlens`` form (the
kernels derive their chunk metadata on the device).
"""

from functools import lru_cache

import torch

from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.triton_utils import tl, triton

# Shapes compiled into FlashInfer's Cake SSD programs.
FLASHINFER_SSD_HEAD_DIM = 64
FLASHINFER_SSD_STATE_SIZE = 128


@lru_cache(maxsize=16)
def _ssd_runner(
    device_index: int,
    stream_id: int,
    nheads: int,
    ngroups: int,
    state_dtype: torch.dtype,
    has_initial_states: bool,
):
    # Runner workspaces are mutable, so like flashinfer.mamba.ssd_combined_fwd
    # keep one runner per stream.
    from flashinfer.mamba import SSDCombined

    with torch.accelerator.device_index(device_index):
        return SSDCombined(
            chunk_size=128,
            nheads=nheads,
            headdim=FLASHINFER_SSD_HEAD_DIM,
            dstate=FLASHINFER_SSD_STATE_SIZE,
            ngroups=ngroups,
            io_dtype=torch.bfloat16,
            state_dtype=state_dtype,
            has_d=True,
            d_has_hdim=False,
            has_initial_states=has_initial_states,
            has_varlen=True,
            has_z=False,
            backend="cake",
        )


def mamba_chunk_scan_flashinfer(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_states: torch.Tensor | None,
) -> None:
    """Scan the prefill tokens and write each sequence's final state.

    x: (seqlen, nheads, headdim) bf16, may be a strided view
    dt: (seqlen, nheads)
    A, dt_bias: (nheads,) fp32
    B, C: (seqlen, ngroups, dstate) bf16, may be strided views
    D: (nheads,) bf16
    cu_seqlens: (batch + 1,) int32, starting at 0
    out: (seqlen, nheads, headdim) bf16, contiguous; written in place
    ssm_state: (num_slots, nheads, headdim, dstate) state cache
    state_indices: (batch,) cache slot of each sequence
    has_initial_states: (batch,) bool, or None when no sequence has a state
    """
    initial_states = (
        gather_initial_states(ssm_state, state_indices, has_initial_states)
        if has_initial_states is not None
        else None
    )
    runner = _ssd_runner(
        x.device.index,
        torch.accelerator.current_stream(x.device).stream_id,
        x.size(1),
        B.size(1),
        ssm_state.dtype,
        initial_states is not None,
    )
    _, final_states = runner.run(
        x.unsqueeze(0),
        dt.unsqueeze(0),
        A,
        B.unsqueeze(0),
        C.unsqueeze(0),
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial_states,
        cu_seqlens=cu_seqlens,
        out=out.unsqueeze(0),
        return_final_states=True,
    )
    scatter_states(ssm_state, final_states, state_indices)


# Static-shape launch of a piecewise-captured mixer
# (MambaConfig.mixed_batch_cudagraph): programs along the sequence axis of the
# state gather / scatter, which stride over the real prefill sequences.
_STATIC_STATE_COPY_SEQS = 16


@lru_cache(maxsize=16)
def _static_ssd_runner(
    device_index: int, nheads: int, ngroups: int, state_dtype: torch.dtype
):
    # The captured graphs replay on one stream, so one runner per device; the
    # mixer's warmup creates it (and compiles its programs) before any capture.
    from flashinfer.mamba import SSDCombined

    with torch.accelerator.device_index(device_index):
        return SSDCombined(
            chunk_size=128,
            nheads=nheads,
            headdim=FLASHINFER_SSD_HEAD_DIM,
            dstate=FLASHINFER_SSD_STATE_SIZE,
            ngroups=ngroups,
            io_dtype=torch.bfloat16,
            state_dtype=state_dtype,
            has_d=True,
            d_has_hdim=False,
            has_initial_states=True,
            has_varlen=True,
            has_z=False,
            backend="cake",
        )


@lru_cache(maxsize=4)
def static_ssd_initial_states(
    device_index: int, rows: int, nheads: int, state_dtype: torch.dtype
) -> torch.Tensor:
    """Zeroed initial states of the static launch, shared by all layers. Each
    call writes only the rows of the real prefill sequences; the rows of the
    decode and padding pseudo-sequences stay zero. Allocate it outside CUDA
    graph capture (the warmup does), where the zero fill runs."""
    return torch.zeros(
        rows,
        nheads,
        FLASHINFER_SSD_HEAD_DIM,
        FLASHINFER_SSD_STATE_SIZE,
        dtype=state_dtype,
        device=torch.device("cuda", device_index),
    )


@triton.jit
def _copy_static_states_kernel(
    src_ptr,
    dst_ptr,
    indices_ptr,
    has_initial_states_ptr,
    num_seqs_ptr,
    stride_cache_slot,
    row_size: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sequence b of the static batch is row b + 1 of the launch's (R + 2)
    state rows. GATHER: cache -> rows (zero without an initial state);
    else rows -> cache (skipping negative slots)."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size
    num_seqs = tl.load(num_seqs_ptr)
    for b in tl.range(tl.program_id(1), num_seqs, tl.num_programs(1), num_stages=1):
        slot = tl.load(indices_ptr + b).to(tl.int64)
        row = (b + 1).to(tl.int64) * row_size + offsets
        if GATHER:
            has_initial_state = (tl.load(has_initial_states_ptr + b) != 0) & (slot >= 0)
            values = tl.load(
                src_ptr
                + tl.where(has_initial_state, slot, 0) * stride_cache_slot
                + offsets,
                mask=mask & has_initial_state,
                other=0.0,
            )
            tl.store(dst_ptr + row, values, mask=mask)
        else:
            values = tl.load(src_ptr + row, mask=mask & (slot >= 0))
            tl.store(
                dst_ptr + slot * stride_cache_slot + offsets,
                values,
                mask=mask & (slot >= 0),
            )


def _copy_static_states(
    cache: torch.Tensor,
    rows: torch.Tensor,
    indices: torch.Tensor,
    has_initial_states: torch.Tensor | None,
    num_seqs: torch.Tensor,
) -> None:
    row_size = cache[0].numel()
    assert cache[0].is_contiguous() and rows.is_contiguous()
    assert rows[0].numel() == row_size and rows.dtype == cache.dtype
    gather = has_initial_states is not None
    block_size = 1024
    grid = (
        triton.cdiv(row_size, block_size),
        min(indices.numel(), _STATIC_STATE_COPY_SEQS),
    )
    _copy_static_states_kernel[grid](
        cache if gather else rows,
        rows if gather else cache,
        indices,
        has_initial_states if gather else indices,
        num_seqs,
        cache.stride(0),
        row_size=row_size,
        GATHER=gather,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )


# Programs of the pseudo-sequence row clearing, which stride over the decode
# and padding rows.
_STATIC_PSEUDO_ROW_PROGRAMS = 128


@triton.jit
def _clear_static_pseudo_rows_kernel(
    x_ptr,
    b_ptr,
    c_ptr,
    cu_seqlens_ptr,
    stride_x,
    stride_b,
    stride_c,
    num_tokens,
    padding_seq,
    x_size: tl.constexpr,
    bc_size: tl.constexpr,
    X_BLOCK: tl.constexpr,
    BC_BLOCK: tl.constexpr,
):
    """Zero the rows of x, B and C outside the prefill sequences: those of the
    decode pseudo-sequence [0, cu_seqlens[1]) and of the padding
    pseudo-sequence [cu_seqlens[padding_seq], num_tokens)."""
    num_decode_tokens = tl.load(cu_seqlens_ptr + 1)
    padding_start = tl.load(cu_seqlens_ptr + padding_seq)
    num_rows = num_decode_tokens + num_tokens - padding_start
    x_offsets = tl.arange(0, X_BLOCK)
    bc_offsets = tl.arange(0, BC_BLOCK)
    x_zeros = tl.zeros([X_BLOCK], dtype=x_ptr.dtype.element_ty)
    bc_zeros = tl.zeros([BC_BLOCK], dtype=b_ptr.dtype.element_ty)
    for r in tl.range(tl.program_id(0), num_rows, tl.num_programs(0), num_stages=1):
        row = tl.where(
            r < num_decode_tokens, r, padding_start + r - num_decode_tokens
        ).to(tl.int64)
        tl.store(x_ptr + row * stride_x + x_offsets, x_zeros, mask=x_offsets < x_size)
        tl.store(
            b_ptr + row * stride_b + bc_offsets, bc_zeros, mask=bc_offsets < bc_size
        )
        tl.store(
            c_ptr + row * stride_c + bc_offsets, bc_zeros, mask=bc_offsets < bc_size
        )


def _clear_static_pseudo_rows(
    x: torch.Tensor, B: torch.Tensor, C: torch.Tensor, cu_seqlens: torch.Tensor
) -> None:
    num_tokens = x.size(0)
    for t in (x, B, C):
        assert t[0].is_contiguous()
    assert B.stride(0) == C.stride(0) and B[0].numel() == C[0].numel()
    grid = (min(num_tokens, _STATIC_PSEUDO_ROW_PROGRAMS),)
    _clear_static_pseudo_rows_kernel[grid](
        x,
        B,
        C,
        cu_seqlens,
        x.stride(0),
        B.stride(0),
        C.stride(0),
        num_tokens,
        cu_seqlens.numel() - 2,
        x_size=x[0].numel(),
        bc_size=B[0].numel(),
        X_BLOCK=triton.next_power_of_2(x[0].numel()),
        BC_BLOCK=triton.next_power_of_2(B[0].numel()),
        num_warps=4,
    )


def mamba_chunk_scan_flashinfer_static(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_states: torch.Tensor,
    num_seqs: torch.Tensor,
) -> None:
    """FlashInfer's scan over the whole static batch of a piecewise-captured
    mixer, [decode tokens | prefill tokens | padding] (T tokens), with every
    shape fixed by T and no host synchronization (the kernels derive their
    segments from cu_seqlens on the device).

    cu_seqlens: (R + 3,) int32, [0, prefill starts..., prefill end, T]:
        sequence 0 covers the decode tokens, 1..R the R prefill slots (empty
        past the real ones) and R + 1 the padding. The pseudo-sequences start
        from zero and are written to out but never to the cache; the decode
        sub-path overwrites the decode rows afterwards.
    state_indices, has_initial_states: (R,) int32 of the prefill slots.
    num_seqs: (1,) int32, the real prefill sequences.
    Other arguments as mamba_chunk_scan_flashinfer.

    The rows of x, B and C outside the prefill sequences are zeroed in place
    first. FlashInfer computes whole 128-token tiles, masking the pairs across
    sequence boundaries, so a NaN or Inf in a pseudo-sequence's rows would
    reach the outputs and the state of the prefill sequence sharing its tile
    (0 * NaN); the conv1d of the mixer leaves those rows unwritten.
    """
    num_slots = state_indices.numel()
    assert cu_seqlens.numel() == num_slots + 3
    nheads, ngroups = x.size(1), B.size(1)
    device_index = x.device.index
    _clear_static_pseudo_rows(x, B, C, cu_seqlens)
    initial_states = static_ssd_initial_states(
        device_index, num_slots + 2, nheads, ssm_state.dtype
    )
    _copy_static_states(
        ssm_state, initial_states, state_indices, has_initial_states, num_seqs
    )
    runner = _static_ssd_runner(device_index, nheads, ngroups, ssm_state.dtype)
    # The runner keeps the workspace of its last call shape (index tables its
    # device kernels fill) and rebuilds it only when the shape changes. Every
    # captured graph must record that fill, so drop the cached workspace.
    cake = getattr(runner, "_cake_runner", None)
    if cake is not None and hasattr(cake, "_workspace_key"):
        cake._workspace_key = None
    _, final_states = runner.run(
        x.unsqueeze(0),
        dt.unsqueeze(0),
        A,
        B.unsqueeze(0),
        C.unsqueeze(0),
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial_states,
        cu_seqlens=cu_seqlens,
        out=out.unsqueeze(0),
        return_final_states=True,
    )
    _copy_static_states(ssm_state, final_states, state_indices, None, num_seqs)
