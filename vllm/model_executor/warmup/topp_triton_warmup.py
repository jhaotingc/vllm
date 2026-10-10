# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile the top-p-only Triton sampling kernels at startup."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


@torch.inference_mode()
def topp_triton_warmup(worker: "Worker") -> None:
    """Compile the top-p kernels (no top-k) for every batch-size class.

    The registered top-k/top-p JIT warmups cover every variant and are skipped
    on CUDA by Model Runner V2 for their startup cost, so requests that sample
    with top_p (e.g. from the model's generation config) compiled these
    kernels on the first sampled steps. Batches of up to _SPLIT_MAX_BATCH rows
    use split-row kernels specialized on the split count; larger batches use
    one kernel.
    """
    if not (worker.use_v2_model_runner and current_platform.is_cuda()):
        return
    if getattr(worker.model_runner, "sampler", None) is None:
        return
    from vllm.v1.sample.ops.topk_topp_triton import (
        _SPLIT_MAX_BATCH,
        _max_sampler_batch_size,
        _topp_split_count,
        apply_top_k_top_p_triton,
        num_compute_units,
    )

    vllm_config = worker.vllm_config
    device = worker.model_runner.device
    vocab_size = vllm_config.model_config.get_vocab_size()
    max_batch_size = _max_sampler_batch_size(vllm_config)
    num_sm = num_compute_units(device.index)
    batch_sizes: dict[int, int] = {}
    for batch_size in range(1, min(max_batch_size, _SPLIT_MAX_BATCH) + 1):
        batch_sizes.setdefault(_topp_split_count(batch_size, num_sm), batch_size)
    sizes = sorted(batch_sizes.values())
    if max_batch_size > _SPLIT_MAX_BATCH:
        sizes.append(_SPLIT_MAX_BATCH + 1)
    for batch_size in sizes:
        logits = torch.randn(batch_size, vocab_size, device=device)
        top_p = torch.full((batch_size,), 0.9, device=device)
        apply_top_k_top_p_triton(logits, None, top_p)
    torch.accelerator.synchronize()
    logger.info("Warmed top-p sampling kernels for batch sizes %s.", sizes)
