# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.mamba.ops.ssd_graph import SSDGraphCache
from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata


@pytest.fixture(scope="module", autouse=True)
def ssd_graph_native(tmp_path_factory):
    """Build the pointer rebinder when a prebuilt helper was not supplied."""
    if not torch.cuda.is_available():
        pytest.skip("SSD graph tests require CUDA")
    native_path = os.environ.get("NANO35_SSD_GRAPH_NATIVE")
    if native_path is None:
        from tools.build_ssd_graph_native import build_ssd_graph_native

        native_path = build_ssd_graph_native(tmp_path_factory.mktemp("ssd_graph"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NANO35_SSD_GRAPH_NATIVE", native_path)
        patch.setenv("NANO35_ENABLE_SSD_GRAPH", "1")
        yield


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("intermediate", [False, True])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("strided", [False, True])
def test_ssd_graph_rebinding(state_dtype, intermediate, chunk_size, strided):
    cache = SSDGraphCache()
    retained: list[tuple[torch.Tensor, torch.Tensor]] = []
    nheads, ngroups, headdim, dstate = 8, 2, 16, 32
    previous = None
    cases = (
        (7, chunk_size + 1),
        (11, chunk_size + 3),
        (13, chunk_size + 7),
        (chunk_size, 35, 19),
        (chunk_size - 3, 31, 17),
        (3,),
        (5,),
        (7, chunk_size + 1),
        (9, chunk_size + 5),
    )
    for iteration, lengths in enumerate(cases):
        torch.manual_seed(iteration + 41)
        rows = sum(lengths)
        if strided:
            width = nheads * headdim + nheads + 2 * ngroups * dstate
            projection = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            x, dt, B, C = projection.split(
                (nheads * headdim, nheads, ngroups * dstate, ngroups * dstate), dim=-1
            )
            x = x.view(rows, nheads, headdim)
            B, C = B.view(rows, ngroups, dstate), C.view(rows, ngroups, dstate)
        else:
            x = torch.randn(rows, nheads, headdim, device="cuda", dtype=torch.bfloat16)
            dt = torch.randn(rows, nheads, device="cuda", dtype=torch.bfloat16)
            B = torch.randn(rows, ngroups, dstate, device="cuda", dtype=torch.bfloat16)
            C = torch.randn_like(B)
        dt = dt - 4
        A = -torch.exp(torch.rand(nheads, device="cuda", dtype=torch.float32))
        D = torch.randn(nheads, device="cuda", dtype=torch.float32)
        bias = torch.randn(nheads, device="cuda", dtype=torch.float32)
        z = torch.randn_like(x)
        cu = (
            torch.tensor((0, *lengths), device="cuda", dtype=torch.int32)
            .cumsum(0)
            .to(torch.int32)
        )
        chunks, last, seq = compute_varlen_chunk_metadata(cu, chunk_size)
        initial = torch.randn(
            len(lengths), nheads, headdim, dstate, device="cuda", dtype=state_dtype
        )
        if previous is not None:
            count = min(initial.shape[0], previous.shape[0])
            initial[:count].copy_(previous[:count])
        options = dict(
            cu_seqlens=cu,
            cu_chunk_seqlens=chunks,
            last_chunk_indices=last,
            seq_idx=seq,
            initial_states=initial,
            D=D,
            z=z,
            dt_bias=bias,
            dt_softplus=True,
            state_dtype=state_dtype,
            return_intermediate_states=intermediate,
        )
        expected_out = torch.empty_like(x)
        actual_out = torch.full_like(x, float("nan"))
        expected = mamba_chunk_scan_combined_varlen(
            x, dt, A, B, C, chunk_size, out=expected_out, **options
        )
        actual = mamba_chunk_scan_combined_varlen(
            x, dt, A, B, C, chunk_size, out=actual_out, graph_cache=cache, **options
        )
        torch.testing.assert_close(actual_out, expected_out, atol=0, rtol=0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            actual_out.contiguous().view(torch.uint8),
            expected_out.contiguous().view(torch.uint8),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
            atol=0,
            rtol=0,
        )
        previous = (actual[last] if intermediate else actual).clone()
        for result, saved in retained:
            torch.testing.assert_close(result, saved, atol=0, rtol=0)
        if not intermediate:
            retained.append((actual, actual.clone()))
    summary = cache.summary()
    assert summary["hits"] >= 3, summary
    assert not summary["errors"], summary
    assert summary["estimated_bytes"] <= cache.max_bytes
    fallback = SSDGraphCache(max_bytes=0)
    fallback_out = torch.full_like(x, float("nan"))
    fallback_state = mamba_chunk_scan_combined_varlen(
        x, dt, A, B, C, chunk_size, out=fallback_out, graph_cache=fallback, **options
    )
    torch.testing.assert_close(fallback_out, expected_out, atol=0, rtol=0)
    torch.testing.assert_close(fallback_state, expected, atol=0, rtol=0)
    assert not fallback.entries and fallback.fallbacks == 1

    # A cache migrated to another stream must fall back, preserving ownership.
    alternate = torch.cuda.Stream()
    alternate.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(alternate):
        alternate_out = torch.full_like(x, float("nan"))
        alternate_state = mamba_chunk_scan_combined_varlen(
            x, dt, A, B, C, chunk_size, out=alternate_out, graph_cache=cache, **options
        )
    torch.cuda.current_stream().wait_stream(alternate)
    torch.testing.assert_close(alternate_out, expected_out, atol=0, rtol=0)
    torch.testing.assert_close(alternate_state, expected, atol=0, rtol=0)


def test_ssd_graph_production_dimensions_and_nested_capture():
    cache = SSDGraphCache()
    nheads, ngroups, headdim, dstate, chunk_size = 64, 8, 64, 128, 128
    for iteration, lengths in enumerate(((7, 129), (11, 131), (13, 133))):
        torch.manual_seed(220 + iteration)
        rows = sum(lengths)
        x = torch.randn(rows, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        dt = torch.randn(rows, nheads, device="cuda", dtype=torch.bfloat16) - 4
        A = -torch.exp(torch.rand(nheads, device="cuda", dtype=torch.float32))
        B = torch.randn(rows, ngroups, dstate, device="cuda", dtype=torch.bfloat16)
        C = torch.randn_like(B)
        bias = torch.randn(nheads, device="cuda", dtype=torch.float32)
        D = torch.randn(nheads, device="cuda", dtype=torch.float32)
        cu = (
            torch.tensor((0, *lengths), device="cuda", dtype=torch.int32)
            .cumsum(0)
            .to(torch.int32)
        )
        chunks, last, seq = compute_varlen_chunk_metadata(cu, chunk_size)
        options = dict(
            cu_seqlens=cu,
            cu_chunk_seqlens=chunks,
            last_chunk_indices=last,
            seq_idx=seq,
            dt_bias=bias,
            D=D,
            dt_softplus=True,
            state_dtype=torch.float16,
        )
        expected_out, actual_out = torch.empty_like(x), torch.full_like(x, float("nan"))
        expected = mamba_chunk_scan_combined_varlen(
            x, dt, A, B, C, chunk_size, out=expected_out, **options
        )
        actual = mamba_chunk_scan_combined_varlen(
            x, dt, A, B, C, chunk_size, out=actual_out, graph_cache=cache, **options
        )
        torch.testing.assert_close(actual_out, expected_out, atol=0, rtol=0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            actual_out.contiguous().view(torch.uint8),
            expected_out.contiguous().view(torch.uint8),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
            atol=0,
            rtol=0,
        )
    assert cache.hits == 2, cache.summary()
    from vllm.model_executor.layers.mamba.ops.ssd_graph import get_ssd_graph_cache

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert get_ssd_graph_cache(x) is None
        graph_out = torch.empty_like(x)
        graph_states = mamba_chunk_scan_combined_varlen(
            x, dt, A, B, C, chunk_size, out=graph_out, **options
        )
    graph.replay()
    torch.testing.assert_close(graph_out, expected_out, atol=0, rtol=0)
    torch.testing.assert_close(graph_states, expected, atol=0, rtol=0)
