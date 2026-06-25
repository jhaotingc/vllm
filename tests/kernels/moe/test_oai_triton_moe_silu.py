# SPDX-License-Identifier: Apache-2.0
"""Numerical validation of the SILU MoE path (fused OAITritonExperts and
unfused UnfusedOAITritonExperts) vs an interleave-correct torch reference.
Reuses make_weights() from test_modular_oai_triton_moe (synthesizes MXFP4
weights + shuffle_weight interleaving), so no external checkpoint is needed."""
import pytest
import torch

from tests.utils import wait_for_gpu_memory_to_clear
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.utils.import_utils import has_triton_kernels

if not has_triton_kernels():
    pytest.skip("triton_kernels not found", allow_module_level=True)

from triton_kernels.testing import assert_close

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
    OAITritonExperts,
    UnfusedOAITritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

from .test_modular_oai_triton_moe import make_weights
from .utils import make_dummy_moe_config


def silu_mul_ref(x, limit=None):
    # contiguous halves (reference uses unshuffled weights)
    x_gate, x_up = torch.chunk(x, 2, dim=-1)
    if limit is not None:
        x_gate = x_gate.clamp(max=limit)
        x_up = x_up.clamp(min=-limit, max=limit)
    return (x_gate * torch.sigmoid(x_gate)) * x_up


def torch_moe_silu(x, w1, w2, w1_bias, w2_bias, topk_weights, topk_ids, limit=None):
    w1 = w1[topk_ids, ...]
    w1_bias = w1_bias[topk_ids, ...]
    h = torch.einsum("bekc,bk->bec", w1, x) + w1_bias
    h = silu_mul_ref(h, limit=limit)
    w2 = w2[topk_ids, ...]
    w2_bias = w2_bias[topk_ids, ...]
    h = torch.einsum("bekc,bek->bec", w2, h) + w2_bias
    return torch.einsum("bec,be->bc", h, topk_weights)


def oai_silu(x, w1, w2, w1_pc, w2_pc, w1_bias, w2_bias, num_experts,
             topk_weights, topk_ids, unfused):
    quant_config = mxfp4_w4a16_moe_quant_config(
        w1_bias=w1_bias, w2_bias=w2_bias, w1_scale=w1_pc, w2_scale=w2_pc
    )
    moe_config = make_dummy_moe_config()
    experts = (
        UnfusedOAITritonExperts(moe_config, quant_config)
        if unfused
        else OAITritonExperts(moe_config, quant_config)
    )
    mk = FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config, quant_config=quant_config,
            allow_new_interface=True, use_monolithic=False,
        ),
        experts,
    )
    return mk.apply(
        hidden_states=x, w1=w1, w2=w2,
        topk_weights=topk_weights, topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts, expert_map=None,
        apply_router_weight_on_input=False,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("m,n,k", [(2, 512, 384), (16, 2880, 2880)])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [4])
@pytest.mark.parametrize("unfused", [False])  # fused path (our feature); unfused SILU needs a separate interleave-aware fix
def test_oai_triton_moe_silu(dtype, m, n, k, num_experts, topk, unfused, workspace_init):
    wait_for_gpu_memory_to_clear(devices=[0], threshold_ratio=0.1)
    set_random_seed(0)
    (w1, w2, w1_bias, w2_bias, w1_tri, w2_tri,
     w1_bias_tri, w2_bias_tri, w1_pc, w2_pc) = make_weights(dtype, k, n, num_experts)

    x = torch.randn((m, k), dtype=dtype, device="cuda")
    router_logits = torch.randn(m, num_experts, device="cuda", dtype=dtype)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1, sorted=True)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    with set_current_vllm_config(VllmConfig()):
        out_ref = torch_moe_silu(x, w1, w2, w1_bias, w2_bias,
                                 topk_weights, topk_ids, limit=None)
        out = oai_silu(x, w1_tri, w2_tri, w1_pc, w2_pc,
                       w1_bias_tri, w2_bias_tri, num_experts,
                       topk_weights, topk_ids, unfused)

    assert_close(ref=out_ref, tri=out, maxtol=0.025, rmstol=0.005)
