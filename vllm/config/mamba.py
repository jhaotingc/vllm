# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum, EnumMeta
from typing import Any, Literal, get_args

from pydantic import field_validator

from vllm.config.utils import config


class _MambaBackendEnumMeta(EnumMeta):
    """Metaclass for MambaBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)
        except KeyError:
            valid = ", ".join(cls.__members__.keys())
            raise ValueError(
                f"Unknown Mamba SSU backend: '{name}'. Valid options are: {valid}"
            ) from None


class MambaBackendEnum(Enum, metaclass=_MambaBackendEnumMeta):
    """Enumeration of supported Mamba SSU (selective state update) backends."""

    TRITON = "triton"
    FLASHINFER = "flashinfer"
    CPU = "cpu"


MambaSSUAlgorithm = Literal["auto", "simple", "vertical", "horizontal"]
MambaSSDBackend = Literal["triton", "flashinfer"]


@config
class MambaConfig:
    """Configuration for Mamba SSM backends."""

    backend: MambaBackendEnum = MambaBackendEnum.TRITON
    """Mamba SSU backend to use."""

    enable_stochastic_rounding: bool = False
    """Enable stochastic rounding when writing SSM state to fp16 cache.
    Uses random bits to unbias the rounding error, which can improve
    numerical stability for long sequences."""
    stochastic_rounding_philox_rounds: int = 0
    """Number of Philox PRNG rounds for stochastic rounding random number
    generation. 0 uses the Triton default. Higher values improve randomness
    quality at the cost of compute."""

    ssu_algorithm: MambaSSUAlgorithm | None = None
    """Selective state update algorithm to use with the FlashInfer backend.
    None defaults to FlashInfer's "auto" algorithm. Forced algorithms must
    be supported by FlashInfer for the active GPU, state dtype, and decoding
    mode."""

    ssd_backend: MambaSSDBackend = "triton"
    """Kernel backend for the chunked SSD scan of Mamba2 prefills. "triton"
    runs vLLM's five Triton kernels; "flashinfer" runs FlashInfer's fused SSD
    scan (two launches; datacenter Blackwell, bfloat16 activations, head dim
    64, state size 128). Prefills that store Mamba checkpoints keep the
    Triton scan."""

    mixed_batch_cudagraph: bool = False
    """Capture the Mamba2 mixer (causal conv1d, SSD scan and decode state
    update) inside the piecewise CUDA graphs that run prefill and mixed
    prefill/decode batches, instead of running it eagerly between graph
    pieces. The mixer then uses shapes fixed by the padded token count, with
    its per-step metadata in persistent buffers, so the cost is padded GPU
    work instead of per-launch host time. Requires the V2 model runner,
    `--use-replayssm --mamba-backend flashinfer` and `--mamba-cache-mode
    none` without prefill checkpoints; the captured scan always uses the
    Triton SSD kernels."""

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `backend` enum type from string."""
        if isinstance(value, str):
            return MambaBackendEnum[value.upper()]
        return value

    def validate_ssu_algorithm(self) -> None:
        if self.ssu_algorithm is None:
            return
        valid_algorithms = get_args(MambaSSUAlgorithm)
        if self.ssu_algorithm not in valid_algorithms:
            valid = ", ".join(valid_algorithms)
            raise ValueError(
                f"Unknown Mamba SSU algorithm: '{self.ssu_algorithm}'. "
                f"Valid options are: {valid}"
            )
        if self.backend != MambaBackendEnum.FLASHINFER:
            raise ValueError(
                "Mamba SSU algorithm selection is only supported with the "
                "FlashInfer backend. Please set `--mamba-backend flashinfer`, "
                "or omit `--mamba-ssu-algorithm`."
            )

    def validate_ssd_backend(self) -> None:
        valid_backends = get_args(MambaSSDBackend)
        if self.ssd_backend not in valid_backends:
            valid = ", ".join(valid_backends)
            raise ValueError(
                f"Unknown Mamba SSD backend: '{self.ssd_backend}'. "
                f"Valid options are: {valid}"
            )
        if self.ssd_backend == "flashinfer":
            from vllm.platforms import current_platform

            if not current_platform.is_device_capability_family(100):
                raise ValueError(
                    "--mamba-ssd-backend flashinfer requires a datacenter "
                    "Blackwell GPU (SM100/SM103)."
                )

    def __post_init__(self):
        self.validate_ssu_algorithm()
        self.validate_ssd_backend()
        if self.enable_stochastic_rounding:
            from vllm.platforms import current_platform

            if not current_platform.is_cuda():
                raise ValueError(
                    "Stochastic rounding for Mamba cache is only supported "
                    "on NVIDIA CUDA platforms. Please do not specify  "
                    "`--enable-mamba-cache-stochastic-rounding`."
                )
            if (
                self.backend == MambaBackendEnum.TRITON
                and not current_platform.is_device_capability_family(100)
            ):
                raise ValueError(
                    "Stochastic rounding for Mamba cache with triton backend requires "
                    "compute capability 10.0 (data center Blackwell). The `cvt.rs` "
                    "PTX instruction is not supported on your GPU. Please do not "
                    "specify `--enable-mamba-cache-stochastic-rounding`, "
                    "or set `--mamba-backend flashinfer`."
                )
