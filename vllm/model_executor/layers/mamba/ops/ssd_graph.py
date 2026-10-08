# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import importlib.util
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import torch

_native = None
_local = threading.local()
_all_caches = []


def _load_native():
    global _native
    if _native is None:
        module = Path(os.environ["NANO35_SSD_GRAPH_NATIVE"])
        spec = importlib.util.spec_from_file_location(
            "nano35_ssd_graph_native_v1", module
        )
        assert spec is not None and spec.loader is not None, (
            f"Cannot load SSD graph helper from {module}"
        )
        _native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_native)
    return _native


class SSDGraphCache:
    """Experimental five-kernel SSD graph cache for one thread/CUDA stream.

    Intermediate states have scratch lifetime: consume before the next replay.
    Normal final states remain a fresh gather outside the graph.
    """

    def __init__(self, max_bytes=2 * 1024**3, max_graphs=32):
        self.max_bytes = max_bytes
        self.max_graphs = max_graphs
        self.estimated_bytes = 0
        self.entries = {}
        self.disabled = set()
        self.hits = 0
        self.misses = 0
        self.fallbacks = 0
        self.errors = []
        self.pool = None
        self.capture_stream = None
        self.owner = None
        self.validated_calls = 0

    def run(self, function, x, dt, A, B, C, chunk_size, out, **kwargs):
        if os.environ.get("NANO35_VALIDATE_SSD_GRAPH") == "1":
            expected_out = torch.empty_like(out)
            expected_states = function(
                x, dt, A, B, C, chunk_size, expected_out, **kwargs
            )
            result = self._run(function, x, dt, A, B, C, chunk_size, out, **kwargs)
            torch.testing.assert_close(out, expected_out, atol=0, rtol=0)
            torch.testing.assert_close(result, expected_states, atol=0, rtol=0)
            torch.testing.assert_close(
                out.contiguous().view(torch.uint8),
                expected_out.contiguous().view(torch.uint8),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                result.contiguous().view(torch.uint8),
                expected_states.contiguous().view(torch.uint8),
                atol=0,
                rtol=0,
            )
            self.validated_calls += 1
            print(
                json.dumps(
                    {
                        "event": "ssd_graph_same_input_bitwise_match",
                        "validated_calls": self.validated_calls,
                        "graph_hits": self.hits,
                        "token_rows": x.shape[0],
                        "output_dtype": str(out.dtype),
                        "state_dtype": str(result.dtype),
                    }
                ),
                flush=True,
            )
            return result
        return self._run(function, x, dt, A, B, C, chunk_size, out, **kwargs)

    def _run(self, function, x, dt, A, B, C, chunk_size, out, **kwargs):
        owner = (
            threading.get_ident(),
            x.device.index,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        if self.owner is None:
            self.owner = owner
        elif self.owner != owner:
            self.fallbacks += 1
            return function(x, dt, A, B, C, chunk_size, out, **kwargs)
        native = _load_native()
        arguments = (
            x,
            dt,
            A,
            B,
            C,
            out,
            kwargs.get("D"),
            kwargs.get("z"),
            kwargs.get("dt_bias"),
            kwargs.get("initial_states"),
            kwargs.get("seq_idx"),
            kwargs["cu_seqlens"],
            kwargs["cu_chunk_seqlens"],
            kwargs["last_chunk_indices"],
        )
        try:
            signature = native.signature(arguments)
        except RuntimeError:
            self.fallbacks += 1
            return function(x, dt, A, B, C, chunk_size, out, **kwargs)
        state_dtype = kwargs.get("state_dtype") or C.dtype
        key = (
            signature,
            chunk_size,
            state_dtype,
            kwargs.get("dt_softplus", False),
            tuple(kwargs.get("dt_limit", (0.0, float("inf")))),
        )
        entry = self.entries.get(key)
        if entry is not None:
            graph, rebinder, states = entry
            rebinder.replay(arguments, torch.cuda.current_stream(x.device).cuda_stream)
            self.hits += 1
            return (
                states
                if kwargs.get("return_intermediate_states", False)
                else states[arguments[-1]]
            )

        self.misses += 1
        result = function(x, dt, A, B, C, chunk_size, out, **kwargs)
        nchunks = arguments[12].numel() - 1
        nheads, headdim = x.shape[1:]
        ngroups, dstate = B.shape[1:]
        state_bytes = torch._utils._element_size(state_dtype)
        estimate = nchunks * (
            nheads * headdim * dstate * (4 + state_bytes)
            + 8 * nheads * chunk_size
            + 4 * ngroups * chunk_size**2
        )
        if (
            key in self.disabled
            or len(self.entries) >= self.max_graphs
            or estimate > 512 * 1024**2
            or self.estimated_bytes + estimate > self.max_bytes
        ):
            self.fallbacks += 1
            return result

        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
            self.capture_stream = torch.cuda.Stream(device=x.device)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        capture_kwargs = dict(kwargs, return_intermediate_states=True)
        # The eager call above warms the exact Triton launch specialization.
        # Manual capture avoids the synchronize in torch.cuda.graph.__enter__.
        with torch.cuda.stream(self.capture_stream):
            graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
            try:
                states = function(x, dt, A, B, C, chunk_size, out, **capture_kwargs)
            finally:
                graph.capture_end()
        graph.instantiate()
        try:
            rebinder = native.GraphRebinder(
                graph.raw_cuda_graph(), graph.raw_cuda_graph_exec(), arguments
            )
        except RuntimeError as error:
            self.disabled.add(key)
            self.errors.append(str(error))
            self.fallbacks += 1
            return result
        self.entries[key] = (graph, rebinder, states)
        self.estimated_bytes += estimate
        print(
            json.dumps(
                {
                    "event": "ssd_graph_capture",
                    "graphs": len(self.entries),
                    "nchunks": nchunks,
                    "estimated_bytes": self.estimated_bytes,
                }
            ),
            flush=True,
        )
        return result

    def summary(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "fallbacks": self.fallbacks,
            "graphs": len(self.entries),
            "estimated_bytes": self.estimated_bytes,
            "errors": self.errors,
            "validated_calls": self.validated_calls,
            "relocations": [r.relocation_counts() for _, r, _ in self.entries.values()],
        }


def get_ssd_graph_cache(tensor):
    if (
        os.environ.get("NANO35_ENABLE_SSD_GRAPH") != "1"
        or not tensor.is_cuda
        or getattr(_local, "suspended", False)
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    stream = torch.cuda.current_stream(tensor.device).cuda_stream
    key = (tensor.device.index, stream)
    pools = getattr(_local, "pools", None)
    if pools is None:
        pools = _local.pools = {}
    if key not in pools:
        pools[key] = SSDGraphCache()
        _all_caches.append((key, pools[key]))
    return pools[key]


@contextmanager
def suspend_ssd_graph_cache():
    previous = getattr(_local, "suspended", False)
    _local.suspended = True
    try:
        yield
    finally:
        _local.suspended = previous


def _dump_usage():
    run = os.environ.get("NANO_RUN")
    if run and _all_caches:
        result = [
            {"device_stream": key, **cache.summary()} for key, cache in _all_caches
        ]
        path = Path(run) / "server" / f"ssd_graph_usage_{os.getpid()}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")


atexit.register(_dump_usage)
