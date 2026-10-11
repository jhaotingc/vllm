# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py

# ruff: noqa: E501

import torch

from vllm.model_executor.layers.mamba.ops.triton_helpers import (
    fast_exp,
    launch_autotuned,
)
from vllm.triton_utils import tl, triton

from .mamba_ssm import softplus


@triton.jit
def _chunk_cumsum_fwd_body(
    pid_c,
    pid_h,
    # Pointers to matrices
    dt_ptr,
    A_ptr,
    dt_bias_ptr,
    dt_out_ptr,
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimension
    nheads: tl.constexpr,
    chunk_size: tl.constexpr,
    dt_min: tl.constexpr,
    dt_max: tl.constexpr,
    # Strides
    stride_dt_seqlen,
    stride_dt_head: tl.constexpr,
    stride_A_head: tl.constexpr,
    stride_dt_bias_head: tl.constexpr,
    stride_dt_out_head,
    stride_dt_out_chunk,
    stride_dt_out_csize: tl.constexpr,
    stride_dA_cs_head,
    stride_dA_cs_chunk,
    stride_dA_cs_csize: tl.constexpr,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
):
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)

    dt_ptr += chunk_seqlen_start * stride_dt_seqlen
    dt_out_ptr += pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (
        offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    )
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (
        offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize
    )
    dA_cs_ptrs = dA_cumsum_ptr + (
        offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize
    )
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    dt = tl.load(
        dt_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(
            dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)

    dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.where(
        (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0
    )
    tl.store(
        dt_out_ptrs,
        dt,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(
        dA_cs_ptrs,
        dA_cs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_H": 2}),
        triton.Config({"BLOCK_SIZE_H": 4}),
        triton.Config({"BLOCK_SIZE_H": 8}),
        triton.Config({"BLOCK_SIZE_H": 16}),
        triton.Config({"BLOCK_SIZE_H": 32}),
        triton.Config({"BLOCK_SIZE_H": 64}),
    ],
    key=["chunk_size", "nheads"],
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr,
    A_ptr,
    dt_bias_ptr,
    dt_out_ptr,
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    num_chunks_ptr,
    # Matrix dimension
    nheads: tl.constexpr,
    chunk_size: tl.constexpr,
    dt_min: tl.constexpr,
    dt_max: tl.constexpr,
    # Strides
    stride_dt_seqlen: tl.int64,
    stride_dt_head: tl.constexpr,
    stride_A_head: tl.constexpr,
    stride_dt_bias_head: tl.constexpr,
    stride_dt_out_head: tl.int64,
    stride_dt_out_chunk: tl.int64,
    stride_dt_out_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    CHUNK_LOOP: tl.constexpr,
    HAS_PAD_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_h = tl.program_id(axis=1)
    if CHUNK_LOOP:
        # Static-shape launch: a fixed number of programs strides over the
        # step's real chunks, whose count is read from the device.
        num_chunks = tl.load(num_chunks_ptr)
        for pid_c in tl.range(
            tl.program_id(axis=0),
            num_chunks,
            tl.num_programs(axis=0),
            num_stages=1,
        ):
            _chunk_cumsum_fwd_body(
                # if dt is long, may cause problems, so use 64 bit
                # https://github.com/triton-lang/triton/issues/1058
                pid_c.to(tl.int64),
                pid_h,
                dt_ptr,
                A_ptr,
                dt_bias_ptr,
                dt_out_ptr,
                dA_cumsum_ptr,
                cu_chunk_seqlens_ptr,
                nheads,
                chunk_size,
                dt_min,
                dt_max,
                stride_dt_seqlen,
                stride_dt_head,
                stride_A_head,
                stride_dt_bias_head,
                stride_dt_out_head,
                stride_dt_out_chunk,
                stride_dt_out_csize,
                stride_dA_cs_head,
                stride_dA_cs_chunk,
                stride_dA_cs_csize,
                DT_SOFTPLUS,
                HAS_DT_BIAS,
                BLOCK_SIZE_H,
                BLOCK_SIZE_CHUNK,
            )
    else:
        # Static-shape launches without the chunk loop end with unused
        # chunks, marked by a negative end boundary.
        if (
            HAS_PAD_CHUNKS
            and tl.load(cu_chunk_seqlens_ptr + tl.program_id(axis=0) + 1) < 0
        ):
            return
        _chunk_cumsum_fwd_body(
            tl.program_id(axis=0).to(tl.int64),
            pid_h,
            dt_ptr,
            A_ptr,
            dt_bias_ptr,
            dt_out_ptr,
            dA_cumsum_ptr,
            cu_chunk_seqlens_ptr,
            nheads,
            chunk_size,
            dt_min,
            dt_max,
            stride_dt_seqlen,
            stride_dt_head,
            stride_A_head,
            stride_dt_bias_head,
            stride_dt_out_head,
            stride_dt_out_chunk,
            stride_dt_out_csize,
            stride_dA_cs_head,
            stride_dA_cs_chunk,
            stride_dA_cs_csize,
            DT_SOFTPLUS,
            HAS_DT_BIAS,
            BLOCK_SIZE_H,
            BLOCK_SIZE_CHUNK,
        )


@triton.autotune(
    configs=[
        # Small headdim/dstate configs (hdim<=64, dstate<=128) - increased parallelism
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        # Low register pressure configs for large dstate (dstate=128)
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        # original configs for larger headdim/dstate values
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
    ],
    key=["hdim", "dstate", "chunk_size"],
)
@triton.jit
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    states_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    num_chunks_ptr,
    # Matrix dimensions
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_b_dstate: tl.constexpr,
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    stride_dt_head: tl.int64,
    stride_dt_chunk: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    # Meta-parameters
    CHUNK_LOOP: tl.constexpr,
    HAS_PAD_CHUNKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    if CHUNK_LOOP:
        # Static-shape launch: a fixed number of programs strides over the
        # step's real chunks, whose count is read from the device.
        num_chunks = tl.load(num_chunks_ptr)
        for pid_c in tl.range(
            tl.program_id(axis=1),
            num_chunks,
            tl.num_programs(axis=1),
            num_stages=1,
        ):
            _chunk_state_fwd_body(
                pid_c.to(tl.int64),
                pid_h,
                pid_m,
                pid_n,
                x_ptr,
                b_ptr,
                states_ptr,
                dt_ptr,
                dA_cumsum_ptr,
                cu_chunk_seqlens_ptr,
                hdim,
                dstate,
                chunk_size,
                nheads_ngroups_ratio,
                stride_x_seqlen,
                stride_x_head,
                stride_x_hdim,
                stride_b_seqlen,
                stride_b_head,
                stride_b_dstate,
                stride_states_chunk,
                stride_states_head,
                stride_states_hdim,
                stride_states_dstate,
                stride_dt_head,
                stride_dt_chunk,
                stride_dt_csize,
                stride_dA_cs_head,
                stride_dA_cs_chunk,
                stride_dA_cs_csize,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
            )
    else:
        # Static-shape launches without the chunk loop end with unused
        # chunks, marked by a negative end boundary.
        if (
            HAS_PAD_CHUNKS
            and tl.load(cu_chunk_seqlens_ptr + tl.program_id(axis=1) + 1) < 0
        ):
            return
        _chunk_state_fwd_body(
            tl.program_id(axis=1).to(tl.int64),
            pid_h,
            pid_m,
            pid_n,
            x_ptr,
            b_ptr,
            states_ptr,
            dt_ptr,
            dA_cumsum_ptr,
            cu_chunk_seqlens_ptr,
            hdim,
            dstate,
            chunk_size,
            nheads_ngroups_ratio,
            stride_x_seqlen,
            stride_x_head,
            stride_x_hdim,
            stride_b_seqlen,
            stride_b_head,
            stride_b_dstate,
            stride_states_chunk,
            stride_states_head,
            stride_states_hdim,
            stride_states_dstate,
            stride_dt_head,
            stride_dt_chunk,
            stride_dt_csize,
            stride_dA_cs_head,
            stride_dA_cs_chunk,
            stride_dA_cs_csize,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )


@triton.jit
def _chunk_state_fwd_body(
    pid_c,
    pid_h,
    pid_m,
    pid_n,
    # Pointers to matrices
    x_ptr,
    b_ptr,
    states_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimensions
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim: tl.constexpr,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate: tl.constexpr,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate: tl.constexpr,
    stride_dt_head,
    stride_dt_chunk,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_head,
    stride_dA_cs_chunk,
    stride_dA_cs_csize: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    b_ptr += (
        chunk_seqlen_start * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
        tl.float32
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        scale = fast_exp(tl.minimum(dA_cs_last - dA_cs_k, 0.0)) * dt_k
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)

        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def _chunk_cumsum_fwd(
    dt,
    A,
    chunk_size,
    cu_chunk_seqlens,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    num_chunks=None,
    chunk_grid=None,
    has_pad_chunks=False,
):
    """`num_chunks` (one-element int32 device tensor), if given, is the number
    of real chunks at the front of `cu_chunk_seqlens` (the rest is static
    padding); `chunk_grid` programs then stride over them."""
    seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = cu_chunk_seqlens.shape[0] - 1
    dt_out = torch.empty(
        nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    dA_cumsum = torch.empty(
        nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    grid_c = nchunks if num_chunks is None else chunk_grid
    grid_chunk_cs = lambda META: (grid_c, triton.cdiv(nheads, META["BLOCK_SIZE_H"]))
    launch_autotuned(
        _chunk_cumsum_fwd_kernel,
        grid_chunk_cs,
        (
            chunk_size,
            nheads,
            dt.dtype,
            A.dtype,
            None if dt_bias is None else dt_bias.dtype,
            cu_chunk_seqlens.dtype,
        ),
        dt_ptr=dt,
        A_ptr=A,
        dt_bias_ptr=dt_bias,
        dt_out_ptr=dt_out,
        dA_cumsum_ptr=dA_cumsum,
        cu_chunk_seqlens_ptr=cu_chunk_seqlens,
        num_chunks_ptr=num_chunks,
        nheads=nheads,
        chunk_size=chunk_size,
        dt_min=dt_limit[0],
        dt_max=dt_limit[1],
        stride_dt_seqlen=dt.stride(0),
        stride_dt_head=dt.stride(1),
        stride_A_head=A.stride(0),
        stride_dt_bias_head=dt_bias.stride(0) if dt_bias is not None else 0,
        stride_dt_out_head=dt_out.stride(0),
        stride_dt_out_chunk=dt_out.stride(1),
        stride_dt_out_csize=dt_out.stride(2),
        stride_dA_cs_head=dA_cumsum.stride(0),
        stride_dA_cs_chunk=dA_cumsum.stride(1),
        stride_dA_cs_csize=dA_cumsum.stride(2),
        DT_SOFTPLUS=dt_softplus,
        HAS_DT_BIAS=dt_bias is not None,
        CHUNK_LOOP=num_chunks is not None,
        HAS_PAD_CHUNKS=has_pad_chunks,
        BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
    )
    return dA_cumsum, dt_out


def _chunk_state_fwd(
    B,
    x,
    dt,
    dA_cumsum,
    cu_chunk_seqlens,
    states=None,
    states_in_fp32=True,
    num_chunks=None,
    chunk_grid=None,
    has_pad_chunks=False,
):
    """`num_chunks`/`chunk_grid`: see `_chunk_cumsum_fwd`."""
    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape

    if states is not None:
        assert states.shape == (nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty(
            (nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype
        )

    grid_c = nchunks if num_chunks is None else chunk_grid
    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        grid_c,
        nheads,
    )
    launch_autotuned(
        _chunk_state_fwd_kernel,
        grid,
        (
            headdim,
            dstate,
            chunk_size,
            x.dtype,
            B.dtype,
            states.dtype,
            dt.dtype,
            dA_cumsum.dtype,
            cu_chunk_seqlens.dtype,
        ),
        x_ptr=x,
        b_ptr=B,
        states_ptr=states,
        dt_ptr=dt,
        dA_cumsum_ptr=dA_cumsum,
        cu_chunk_seqlens_ptr=cu_chunk_seqlens,
        num_chunks_ptr=num_chunks,
        hdim=headdim,
        dstate=dstate,
        chunk_size=chunk_size,
        nheads_ngroups_ratio=nheads // ngroups,
        stride_x_seqlen=x.stride(0),
        stride_x_head=x.stride(1),
        stride_x_hdim=x.stride(2),
        stride_b_seqlen=B.stride(0),
        stride_b_head=B.stride(1),
        stride_b_dstate=B.stride(2),
        stride_states_chunk=states.stride(0),
        stride_states_head=states.stride(1),
        stride_states_hdim=states.stride(2),
        stride_states_dstate=states.stride(3),
        stride_dt_head=dt.stride(0),
        stride_dt_chunk=dt.stride(1),
        stride_dt_csize=dt.stride(2),
        stride_dA_cs_head=dA_cumsum.stride(0),
        stride_dA_cs_chunk=dA_cumsum.stride(1),
        stride_dA_cs_csize=dA_cumsum.stride(2),
        CHUNK_LOOP=num_chunks is not None,
        HAS_PAD_CHUNKS=has_pad_chunks,
    )
    return states
