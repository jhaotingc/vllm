# Experimental SSD prefill graph cache

This opt-in cache reduces CPU allocation and dispatch work for Mamba2 prefill by
capturing the existing five SSD Triton kernels and rebinding their live tensor
pointers before replay. ReplaySSM decode remains a separate path. Shapes, strides,
FP32 intermediates and the configured persistent state dtype are preserved.

Each cache belongs to one thread, device and CUDA stream. Its fixed limits are
32 graphs, 2 GiB of estimated intermediate storage and 512 MiB per entry.
Unsupported layouts, exhausted budgets and vLLM model graph capture use the
existing eager SSD path. Intermediate states have scratch lifetime and must be
consumed before another replay; final states use a fresh gather outside the graph.

The native helper uses CUDA driver kernel-parameter introspection. The measured
environment was CUDA 13.4 with PyTorch 2.15 nightly on GB300. Build the helper in
the same PyTorch environment used to serve the model:

```bash
.venv/bin/python tools/build_ssd_graph_native.py --build-dir /path/to/ssd-graph-cache
export NANO35_SSD_GRAPH_NATIVE=/path/to/ssd-graph-cache/nano35_ssd_graph_native_v1.so
export NANO35_ENABLE_SSD_GRAPH=1
```

The feature is disabled by default. Set `NANO35_VALIDATE_SSD_GRAPH=1` only for
correctness runs: it recomputes the eager SSD output and states on the same inputs
and compares them byte for byte. Run the kernel suite with:

```bash
.venv/bin/python -m pytest tests/kernels/mamba/test_ssd_graph.py
```

The suite builds the helper when its environment variable is absent. Optional
`NANO_RUN` usage statistics are written to an existing `server/` directory beneath
that path.

Cadence8 is selectable with
`--scheduler-cls vllm.v1.core.sched.nano35_cadence_scheduler.Nano35Cadence8Scheduler`.
It uses the existing prefill throttle policy, including its capacity and
prefill-only progress rules.

This remains an experimental performance option. With ReplaySSM, the original
cache improved the measured c16/c32 workloads but regressed c128 throughput by
1.61%. Those measurements used an ascending sweep and do not establish success
for the descending workload specified in the experiment plan.
