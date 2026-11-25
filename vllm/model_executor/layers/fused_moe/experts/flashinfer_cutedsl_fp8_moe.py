# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer CuTe-DSL block-FP8 routed experts for Hopper.

Selected with `--moe-backend flashinfer_cutedsl` for FP8 checkpoints with
128x128 weight blocks on SM90. The kernels quantize activations themselves
(dynamic 1x128, fp32 scales), apply SiLU-GLU and reduce the top-k outputs in
fp32, so they take unquantized inputs and skip vLLM's finalize reduction.
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.platforms import current_platform


@functools.cache
def _sm90_fp8_block_moe() -> ModuleType | None:
    try:
        from flashinfer.fused_moe.cute_dsl import sm90_fp8_block_moe
    except ImportError:
        return None
    return sm90_fp8_block_moe


# One workspace per device, shared by every layer (target and drafter): the MoE
# calls of a forward run one after another on the same stream.
_workspaces: dict[torch.device, Any] = {}


def _shared_workspace(device: torch.device, max_num_tokens: int) -> Any:
    fi = _sm90_fp8_block_moe()
    assert fi is not None
    ws = _workspaces.get(device)
    if ws is None or ws.max_num_tokens < max_num_tokens:
        ws = fi.Sm90Fp8BlockMoEWorkspace(max_num_tokens, device)
        _workspaces[device] = ws
    return ws


def _max_num_tokens() -> int:
    config = get_current_vllm_config()
    tokens = config.scheduler_config.max_num_batched_tokens
    capture = config.compilation_config.max_cudagraph_capture_size
    return max(tokens, capture or 0)


class FlashInferCuteDSLFp8BlockExperts(mk.FusedMoEExpertsModular):
    """Block-FP8 W8A8 routed experts from FlashInfer's SM90 CuTe-DSL kernels."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self._weights: Any = None
        self._workspace: Any = None

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return False, reason
        fi = _sm90_fp8_block_moe()
        assert fi is not None
        shape = (
            moe_config.num_local_experts,
            moe_config.experts_per_token,
            moe_config.hidden_dim,
            moe_config.intermediate_size_per_partition,
        )
        if not fi.is_sm90_fp8_block_moe_supported(*shape):
            return False, (
                "kernel does not support (experts, top_k, hidden, intermediate) "
                f"= {shape}"
            )
        if moe_config.has_bias:
            return False, "kernel does not support expert bias"
        return True, None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability(90)
            and _sm90_fp8_block_moe() is not None
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return moe_parallel_config.ep_size == 1 and moe_parallel_config.dp_size == 1

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (0,), (0,), (M, K)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Re-lay out the expert weights for the kernels.

        `w13_weight`/`w2_weight` keep their `[E, 2I, H]`/`[E, H, I]`
        shapes (the modular kernel reads them) but now view the packed
        small-batch layouts. The large-batch kernel needs a second `w13`
        layout, kept on this object.
        """
        fi = _sm90_fp8_block_moe()
        assert fi is not None
        if layer.apply_router_weight_on_input:
            raise ValueError(
                "flashinfer_cutedsl FP8 MoE does not support "
                "apply_router_weight_on_input"
            )
        for name, scale in (
            ("w1_scale", self.quant_config.w1_scale),
            ("w2_scale", self.quant_config.w2_scale),
        ):
            if (
                scale is None
                or scale.dtype != torch.float32
                or not scale.is_contiguous()
            ):
                raise ValueError(
                    f"flashinfer_cutedsl FP8 MoE needs contiguous float32 {name}"
                )
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        weights = fi.prepare_sm90_fp8_block_moe_weights(w13.data, w2.data)
        w13.data = weights.w13_decode.view(w13.shape)
        w2.data = weights.w2.view(w2.shape)
        self._weights = weights
        self._workspace = _shared_workspace(w13.device, _max_num_tokens())

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        fi = _sm90_fp8_block_moe()
        assert fi is not None and self._weights is not None
        assert expert_map is None and not apply_router_weight_on_input
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.float()
        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None
        fi.cute_dsl_fused_moe_fp8_block_scale_sm90(
            hidden_states.contiguous(),
            topk_ids.contiguous(),
            topk_weights.contiguous(),
            self._weights,
            self.quant_config.w1_scale,
            self.quant_config.w2_scale,
            out=output,
            workspace=self._workspace,
        )
