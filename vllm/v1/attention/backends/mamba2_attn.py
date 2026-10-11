# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointBuilder,
    MambaPrefillCheckpointMetadata,
)
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import MambaSpec

# causal_conv1d_fn processes this many tokens of a sequence per program.
_CONV1D_BLOCK_M = 8
# Programs per (tile, head) along the sequence axis of the static SSD's
# per-sequence kernels, and extra programs (two per such sequence) along the
# chunk axis of its chunk-parallel kernels. Real sequences or chunks beyond
# these are covered by the programs looping, so typical steps (up to this many
# prefills) need one pass while worst-case shapes cost nothing extra.
_STATIC_SSD_LOOP_SEQS = 16
# The chunk-parallel kernels stride over the real chunks only up to this many
# padded tokens. Their strided bodies need more registers, which outweighs the
# saved programs once a step has hundreds of real chunks; larger launches keep
# one program per static chunk, the unused ones exiting at once.
_STATIC_SSD_CHUNK_LOOP_MAX_TOKENS = 8192
# With --mamba-ssd-backend flashinfer, static launches run FlashInfer's scan
# with one sequence slot per this many padded tokens, and at least
# _STATIC_FLASHINFER_MIN_SEQS slots (see mamba2_static_flashinfer_seqs).
_STATIC_FLASHINFER_TOKENS_PER_SEQ = 512
_STATIC_FLASHINFER_MIN_SEQS = 8
# FlashInfer's scan splits the token stream into chunks of this many tokens.
_FLASHINFER_SSD_CHUNK = 128


def compute_varlen_chunk_metadata(
    query_start_loc: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build chunk-aligned, variable-length metadata used by Mamba2 SSD kernels.

    Given per-sequence cumulative token starts `query_start_loc` of shape [B+1]
    and a physical `chunk_size`, returns three tensors on the same device:
      - cu_chunk_seqlens:  (nchunks+1,) int32   exclusive prefix-sum of
        logical-chunk lengths (each logical chunk never crosses a sequence or
        physical-chunk boundary).
      - last_chunk_indices: (B,)       int32   index of the last logical chunk
        for each sequence (=-1 for empty sequences).
      - seq_idx_chunks:     (nchunks,) int32   sequence index for each logical
        chunk in order.

    This is intentionally lightweight and CPU-side; it mirrors the metadata
    produced by the V1 Mamba2 meta-data builder and is exported so tests
    (and other callers) can avoid duplicating the logic.
    """
    assert query_start_loc.ndim == 1, "query_start_loc must be 1-D [B+1]"
    assert int(query_start_loc[0].item()) == 0, "query_start_loc[0] must be 0"
    device = query_start_loc.device

    qsl64 = query_start_loc.to(torch.int64)
    starts = qsl64[:-1].tolist()
    ends = qsl64[1:].tolist()
    total = int(qsl64[-1].item())

    chunk_lens: list[int] = []
    seq_idx_chunks: list[int] = []
    last_chunk_indices: list[int] = [-1] * len(starts)

    for b, (s, e) in enumerate(zip(starts, ends)):
        if e <= s:
            # empty sequence
            continue
        pos = s
        while pos < e:
            # split at both sequence boundaries and physical chunk boundaries
            room = chunk_size - (pos % chunk_size)
            take = min(room, e - pos)
            chunk_lens.append(int(take))
            seq_idx_chunks.append(b)
            last_chunk_indices[b] = len(chunk_lens) - 1
            pos += take

    # Exclusive prefix sum over logical-chunk lengths
    if chunk_lens:
        cu_chunk_seqlens_list = [0] + list(itertools.accumulate(chunk_lens))
        # Final boundary must equal total tokens (check on host to avoid a sync)
        assert cu_chunk_seqlens_list[-1] == total
    else:
        cu_chunk_seqlens_list = [0]
    cu_chunk_seqlens = async_tensor_h2d(
        cu_chunk_seqlens_list, dtype=torch.int32, device=device
    )

    # last_chunk_indices is empty when there are no sequences (len(starts) == 0).
    last_chunk_indices_t = async_tensor_h2d(
        last_chunk_indices, dtype=torch.int32, device=device
    )
    seq_idx_chunks_t = async_tensor_h2d(
        seq_idx_chunks, dtype=torch.int32, device=device
    )
    return cu_chunk_seqlens, last_chunk_indices_t, seq_idx_chunks_t


class Mamba2AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA2_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba2AttentionMetadataBuilder"]:
        return Mamba2AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


def mamba2_static_sizes(
    num_tokens: int, max_num_seqs: int, max_decode_rows: int, chunk_size: int
) -> tuple[int, int, int, int]:
    """Static launch sizes of a Mamba2 mixer for `num_tokens` padded tokens:
    (prefill sequence slots, SSD chunks, causal_conv1d_fn programs, decode
    rows). Each sequence adds at most two partial chunks (one finishing the
    chunk its computed tokens left open, one at its end) and one partial
    conv1d block."""
    num_seqs = min(max_num_seqs, num_tokens)
    num_chunks = cdiv(num_tokens, chunk_size) + 2 * num_seqs
    num_conv_programs = cdiv(num_tokens, _CONV1D_BLOCK_M) + num_seqs
    num_decode_rows = min(max_decode_rows, num_tokens)
    return num_seqs, num_chunks, num_conv_programs, num_decode_rows


def mamba2_static_flashinfer_seqs(num_tokens: int, max_num_seqs: int) -> int:
    """Prefill sequence slots of FlashInfer's static scan for `num_tokens`
    padded tokens. Its kernel runs one work item per (slot, head), empty
    slots included, so the slots follow the step size: enough for prompts
    of about _STATIC_FLASHINFER_TOKENS_PER_SEQ tokens. Steps with more
    prefills run eagerly (the model runner checks this bound on the
    unpadded token count, which can only lower it)."""
    return min(
        max_num_seqs,
        num_tokens,
        max(
            _STATIC_FLASHINFER_MIN_SEQS,
            cdiv(num_tokens, _STATIC_FLASHINFER_TOKENS_PER_SEQ),
        ),
    )


def mamba2_static_ssd_grids(
    num_tokens: int, num_seqs: int, num_chunks: int, chunk_size: int
) -> tuple[int | None, int]:
    """Launch widths of the static SSD kernels: (programs along the chunk
    axis, or None for one program per static chunk; programs along the
    sequence axis). Strided kernels loop over the step's real chunks or
    sequences, so these only set the parallelism."""
    seq_grid = min(num_seqs, _STATIC_SSD_LOOP_SEQS)
    if num_tokens > _STATIC_SSD_CHUNK_LOOP_MAX_TOKENS:
        return None, seq_grid
    chunk_grid = min(
        num_chunks, cdiv(num_tokens, chunk_size) + 2 * _STATIC_SSD_LOOP_SEQS
    )
    return chunk_grid, seq_grid


@dataclass
class Mamba2StaticMetadata:
    """Fixed-shape metadata of a Mamba2 mixer captured in a piecewise CUDA
    graph (MambaConfig.mixed_batch_cudagraph).

    Shapes depend only on the padded token count, so one graph per capture
    size serves every mix of prefills and decodes. All tensors are views of
    persistent buffers that the builder rewrites before each step. Token
    offsets index the padded batch [decode tokens | prefill tokens | padding].
    Padding prefill sequences have state index PAD_SLOT_ID, padding decode
    rows NULL_BLOCK_ID, padding chunks a negative end boundary and padding
    conv1d programs PAD_SLOT_ID.
    """

    num_tokens: int
    # Prefill sequences.
    query_start_loc_p: torch.Tensor  # (num_seqs + 1,) int32
    state_indices_p: torch.Tensor  # (num_seqs,) int32
    has_initial_states_p: torch.Tensor  # (num_seqs,) int32
    cu_chunk_seqlens_p: torch.Tensor  # (num_chunks + 1,) int32
    seq_idx_p: torch.Tensor  # (num_chunks,) int32
    last_chunk_indices_p: torch.Tensor  # (num_seqs,) int32
    # Real prefill sequences and chunks of this step (the rest is padding),
    # and the launch widths of the SSD kernels that stride over them.
    num_seqs_p: torch.Tensor  # (1,) int32
    num_chunks_p: torch.Tensor  # (1,) int32
    # FlashInfer scan of the whole batch (--mamba-ssd-backend flashinfer),
    # see mamba_chunk_scan_flashinfer_static: sequence offsets
    # [0, starts of prefills 2..n, num_tokens, ...] over R slots, and the
    # prefill tokens [start, end).
    cu_seqlens_fi: torch.Tensor | None  # (R + 1,) int32
    prefill_bounds_fi: torch.Tensor | None  # (2,) int32
    num_seqs_fi: torch.Tensor | None  # (1,) int32, the prefills it scans
    seq_grid: int
    chunk_grid: int | None  # None: one program per static chunk
    # causal_conv1d_fn launch metadata (it reads these attribute names).
    nums_dict: dict[int, dict[str, Any]]
    batch_ptr: torch.Tensor  # (num_conv_programs,) int32
    token_chunk_offset_ptr: torch.Tensor  # (num_conv_programs,) int32
    # Decode rows.
    query_start_loc_d: torch.Tensor  # (num_decode_rows + 1,) int32
    state_indices_d: torch.Tensor  # (num_decode_rows, 1 + num_spec) int32
    replayssm_state_indices_d: torch.Tensor | None  # (num_decode_rows,) int32
    num_accepted_tokens: torch.Tensor | None  # (num_decode_rows,) int32
    replayssm_scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None


@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    prep_initial_states: bool = False
    chunk_size: int = 0

    # Chunk-related metadata (only for prefill)
    seq_idx_p: torch.Tensor | None = None

    # Internal prefill checkpoints, one entry per prefill row. The chunk
    # index selects the varlen_states row holding the checkpoint state.
    checkpoint_chunk_idx: torch.Tensor | None = None
    checkpoint_meta: MambaPrefillCheckpointMetadata | None = None

    # Set when the mixer may run inside a piecewise CUDA graph.
    static: Mamba2StaticMetadata | None = None


class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    metadata_cls = Mamba2AttentionMetadata

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        assert chunk_size is not None, (
            "chunk_size needs to be set in the model config for Mamba2 models"
        )
        self.chunk_size: int = chunk_size
        self.checkpoint_builder = MambaPrefillCheckpointBuilder(
            vllm_config, kv_cache_spec
        )
        if self.use_spec_decode:
            # Under spec decode, update_block_table() keeps the source group's
            # batch-level decode buffers (like GDN), so only MRV2, which also
            # reuses metadata at capture, may share metadata across groups.
            self.supports_update_block_table = (
                vllm_config.use_v2_model_runner and device.type == "cuda"
            )

        # Persistent buffers for a mixer captured in piecewise CUDA graphs.
        self.static_max_tokens = 0
        if vllm_config.mamba_config.mixed_batch_cudagraph:
            self.static_max_tokens = (
                self.compilation_config.max_cudagraph_capture_size
                or vllm_config.scheduler_config.max_num_batched_tokens
            )
            num_seqs, num_chunks, num_conv_programs, num_decode_rows = (
                mamba2_static_sizes(
                    self.static_max_tokens,
                    vllm_config.scheduler_config.max_num_seqs,
                    self.decode_cudagraph_max_bs,
                    self.chunk_size,
                )
            )
            # Host-computed metadata goes up in one copy per step.
            segment_sizes = {
                "query_start_loc_p": num_seqs + 1,
                "cu_seqlens_fi": num_seqs + 1,
                "prefill_bounds_fi": 2,
                "num_seqs_fi": 1,
                "has_initial_states_p": num_seqs,
                "cu_chunk_seqlens_p": num_chunks + 1,
                "seq_idx_p": num_chunks,
                "last_chunk_indices_p": num_seqs,
                "num_seqs_p": 1,
                "num_chunks_p": 1,
                "batch_ptr": num_conv_programs,
                "token_chunk_offset_ptr": num_conv_programs,
                "query_start_loc_d": num_decode_rows + 1,
            }
            offsets = itertools.accumulate(segment_sizes.values(), initial=0)
            self._static_offsets = dict(zip(segment_sizes, offsets))
            self._static_host_size = sum(segment_sizes.values())
            self._static_device = torch.empty(
                self._static_host_size, dtype=torch.int32, device=device
            )
            # Per KV cache group, staged by build() and update_block_table().
            self._static_state_indices_p = torch.empty(
                num_seqs, dtype=torch.int32, device=device
            )
            self._static_flashinfer_ssd = (
                vllm_config.mamba_config.ssd_backend == "flashinfer"
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba2AttentionMetadata:
        common = self._compute_common_metadata(
            common_attn_metadata,
            num_accepted_tokens=kwargs.get("num_accepted_tokens"),
            num_decode_draft_tokens_cpu=kwargs.get("num_decode_draft_tokens_cpu"),
        )

        seq_idx_p = None
        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None
        checkpoint_chunk_idx = None
        checkpoint_meta = None
        prep_initial_states = False
        chunk_lists = None

        # Compute seq_idx for prefill only
        if common.num_prefills > 0:
            prep_initial_states = False
            if common.has_initial_states_p is not None:
                # Same condition as `has_initial_states_p`, but derived from CPU
                # data so it needs no D2H. `seq_lens_cpu_upper_bound` is precise
                # for prefill rows, which is all this slice covers.
                num_computed_tokens_p_cpu, _ = self._prefill_cpu_metadata(
                    common_attn_metadata,
                    common.num_reqs,
                    common.num_prefills,
                    common.num_decode_tokens,
                )
                prep_initial_states = bool((num_computed_tokens_p_cpu > 0).any())

            checkpoint_offsets_p = None
            first = common.num_reqs - common.num_prefills
            checkpoint = self.checkpoint_builder.build(
                common_attn_metadata,
                list(range(first, common.num_reqs)),
            )
            if checkpoint is not None:
                # The host offsets place a chunk boundary on the checkpoint.
                checkpoint_offsets_p = checkpoint.offsets
                checkpoint_meta = checkpoint

            (
                cu_chunk_seqlen_p,
                seq_idx_p,
                last_chunk_indices_p,
                checkpoint_chunk_idx,
                chunk_lists,
            ) = self._build_chunk_metadata_tensors(
                self.chunk_size,
                common,
                common_attn_metadata,
                checkpoint_offsets_p,
            )

        metadata = replace(
            common,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            seq_idx_p=seq_idx_p,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
            checkpoint_chunk_idx=checkpoint_chunk_idx,
            checkpoint_meta=checkpoint_meta,
        )
        if self.static_max_tokens and checkpoint_meta is None:
            metadata = replace(
                metadata,
                static=self._build_static(metadata, common_attn_metadata, chunk_lists),
            )
        return metadata

    def _build_static(
        self,
        metadata: Mamba2AttentionMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        chunk_lists: tuple[list[int], list[int], list[int]] | None,
    ) -> Mamba2StaticMetadata | None:
        """Fill the persistent static-shape metadata for this step's padded
        token count (the slot mapping's length). None if the step is larger
        than any captured graph and so always runs eagerly."""
        num_tokens = common_attn_metadata.slot_mapping.shape[0]
        if num_tokens > self.static_max_tokens:
            return None
        num_seqs, num_chunks, num_conv_programs, num_decode_rows = mamba2_static_sizes(
            num_tokens,
            self.vllm_config.scheduler_config.max_num_seqs,
            self.decode_cudagraph_max_bs,
            self.chunk_size,
        )
        chunk_grid, seq_grid = mamba2_static_ssd_grids(
            num_tokens, num_seqs, num_chunks, self.chunk_size
        )
        num_fi_seqs = (
            mamba2_static_flashinfer_seqs(
                num_tokens, self.vllm_config.scheduler_config.max_num_seqs
            )
            if self._static_flashinfer_ssd
            else 0
        )
        num_decodes = metadata.num_decodes
        num_prefills = metadata.num_prefills
        num_decode_tokens = metadata.num_decode_tokens
        num_actual_tokens = num_decode_tokens + metadata.num_prefill_tokens
        assert num_prefills <= num_seqs and num_decodes <= num_decode_rows
        query_start_loc = common_attn_metadata.query_start_loc_cpu.numpy()
        offsets = self._static_offsets
        host = np.empty(self._static_host_size, dtype=np.int32)

        def segment(name: str, size: int) -> np.ndarray:
            return host[offsets[name] : offsets[name] + size]

        # Prefill sequences, in absolute token offsets.
        qsl_p = segment("query_start_loc_p", num_seqs + 1)
        qsl_p[: num_prefills + 1] = query_start_loc[
            num_decodes : num_decodes + num_prefills + 1
        ]
        qsl_p[num_prefills + 1 :] = num_actual_tokens
        if num_fi_seqs:
            # One FlashInfer sequence per prefill: the first also covers the
            # decode rows (identity rows of the scan); the last ends at the
            # first 128-token boundary after the real tokens, and the padding
            # after it belongs to no sequence, so the scan skips it. The empty
            # slots start at that boundary: an empty sequence starting inside
            # a chunk breaks the final state of the sequence before it.
            # The model runner runs steps with more prefills than the slots
            # eagerly, so only metadata that is never replayed (capture dummy
            # batches, profile and eager steps) folds the surplus prefills
            # into the last slot.
            num_fi_prefills = min(num_prefills, num_fi_seqs)
            cu_fi = segment("cu_seqlens_fi", num_fi_seqs + 1)
            cu_fi[0] = 0
            cu_fi[1:num_fi_prefills] = qsl_p[1:num_fi_prefills]
            cu_fi[max(num_fi_prefills, 1) :] = min(
                num_tokens, round_up(num_actual_tokens, _FLASHINFER_SSD_CHUNK)
            )
            bounds = segment("prefill_bounds_fi", 2)
            bounds[0] = qsl_p[0]
            bounds[1] = num_actual_tokens
            segment("num_seqs_fi", 1)[0] = num_fi_prefills
        has_initial_states = segment("has_initial_states_p", num_seqs)
        has_initial_states[:] = 0
        if num_prefills > 0:
            num_computed_tokens_p_cpu, _ = self._prefill_cpu_metadata(
                common_attn_metadata,
                metadata.num_reqs,
                num_prefills,
                num_decode_tokens,
            )
            has_initial_states[:num_prefills] = num_computed_tokens_p_cpu.numpy() > 0
        cu_chunk_seqlens, seq_idx, last_chunk_indices = chunk_lists or ([0], [], [])
        num_real_chunks = len(seq_idx)
        assert num_real_chunks <= num_chunks
        cu_chunks = segment("cu_chunk_seqlens_p", num_chunks + 1)
        cu_chunks[: num_real_chunks + 1] = cu_chunk_seqlens
        cu_chunks[: num_real_chunks + 1] += num_decode_tokens
        cu_chunks[num_real_chunks + 1 :] = -1
        seq_idx_seg = segment("seq_idx_p", num_chunks)
        seq_idx_seg[:num_real_chunks] = seq_idx
        seq_idx_seg[num_real_chunks:] = -1
        last_chunks = segment("last_chunk_indices_p", num_seqs)
        last_chunks[:num_prefills] = last_chunk_indices
        last_chunks[num_prefills:] = num_real_chunks - 1
        segment("num_seqs_p", 1)[0] = num_prefills
        segment("num_chunks_p", 1)[0] = num_real_chunks
        # causal_conv1d_fn: one program per _CONV1D_BLOCK_M tokens of a prefill.
        seqlens = np.diff(qsl_p[: num_prefills + 1])
        programs = -(-seqlens // _CONV1D_BLOCK_M)
        num_real_programs = int(programs.sum())
        assert num_real_programs <= num_conv_programs
        batch_ptr = segment("batch_ptr", num_conv_programs)
        batch_ptr[:num_real_programs] = np.repeat(np.arange(num_prefills), programs)
        batch_ptr[num_real_programs:] = PAD_SLOT_ID
        chunk_offsets = segment("token_chunk_offset_ptr", num_conv_programs)
        chunk_offsets[:num_real_programs] = np.arange(num_real_programs) - np.repeat(
            np.cumsum(programs) - programs, programs
        )
        chunk_offsets[num_real_programs:] = PAD_SLOT_ID
        # Decode rows.
        qsl_d = segment("query_start_loc_d", num_decode_rows + 1)
        qsl_d[: num_decodes + 1] = query_start_loc[: num_decodes + 1]
        qsl_d[num_decodes + 1 :] = num_decode_tokens
        self._static_device.copy_(
            torch.from_numpy(host).pin_memory(), non_blocking=True
        )

        def view(name: str, size: int) -> torch.Tensor:
            return self._static_device[offsets[name] : offsets[name] + size]

        num_accepted_tokens = None
        if self.use_spec_decode:
            num_accepted_tokens = self.decode_num_accepted_tokens[:num_decode_rows]
            if (
                metadata.num_accepted_tokens is not None
                and metadata.num_accepted_tokens.data_ptr()
                != num_accepted_tokens.data_ptr()
            ):
                num_accepted_tokens[:num_decodes].copy_(
                    metadata.num_accepted_tokens[:num_decodes], non_blocking=True
                )
            num_accepted_tokens[num_decodes:] = 1
        replayssm_scratch = None
        if self.decode_replayssm_scratch is not None:
            cb_scaled, cumAdt_vec, cb_old = self.decode_replayssm_scratch
            replayssm_scratch = (
                cb_scaled[:num_decode_rows],
                cumAdt_vec[:num_decode_rows],
                cb_old[:num_decode_rows],
            )
        batch_ptr_view = view("batch_ptr", num_conv_programs)
        chunk_offsets_view = view("token_chunk_offset_ptr", num_conv_programs)
        static = Mamba2StaticMetadata(
            num_tokens=num_tokens,
            query_start_loc_p=view("query_start_loc_p", num_seqs + 1),
            state_indices_p=self._static_state_indices_p[:num_seqs],
            has_initial_states_p=view("has_initial_states_p", num_seqs),
            cu_chunk_seqlens_p=view("cu_chunk_seqlens_p", num_chunks + 1),
            seq_idx_p=view("seq_idx_p", num_chunks),
            last_chunk_indices_p=view("last_chunk_indices_p", num_seqs),
            num_seqs_p=view("num_seqs_p", 1),
            num_chunks_p=view("num_chunks_p", 1),
            cu_seqlens_fi=(
                view("cu_seqlens_fi", num_fi_seqs + 1) if num_fi_seqs else None
            ),
            prefill_bounds_fi=view("prefill_bounds_fi", 2) if num_fi_seqs else None,
            num_seqs_fi=view("num_seqs_fi", 1) if num_fi_seqs else None,
            seq_grid=seq_grid,
            chunk_grid=chunk_grid,
            nums_dict={
                _CONV1D_BLOCK_M: {
                    "tot": num_conv_programs,
                    "batch_ptr": batch_ptr_view,
                    "token_chunk_offset_ptr": chunk_offsets_view,
                    "mlist": None,
                    "mlist_len": 0,
                    "offsetlist": None,
                }
            },
            batch_ptr=batch_ptr_view,
            token_chunk_offset_ptr=chunk_offsets_view,
            query_start_loc_d=view("query_start_loc_d", num_decode_rows + 1),
            state_indices_d=self.state_indices_tensor_d[:num_decode_rows],
            replayssm_state_indices_d=None,
            num_accepted_tokens=num_accepted_tokens,
            replayssm_scratch=replayssm_scratch,
        )
        return self._stage_static_state_indices(
            static,
            metadata,
            metadata.state_indices_tensor_d,
            metadata.state_indices_tensor_p,
        )

    def _stage_static_state_indices(
        self,
        static: Mamba2StaticMetadata,
        metadata: Mamba2AttentionMetadata,
        state_indices_tensor_d: torch.Tensor | None,
        state_indices_tensor_p: torch.Tensor | None,
    ) -> Mamba2StaticMetadata:
        """Stage this KV cache group's state indices into its own persistent
        static buffers (the other static metadata is batch-level)."""
        num_prefills = metadata.num_prefills
        state_indices_p = self._static_state_indices_p[
            : static.state_indices_p.shape[0]
        ]
        if num_prefills > 0:
            assert state_indices_tensor_p is not None
            state_indices_p[:num_prefills].copy_(
                state_indices_tensor_p, non_blocking=True
            )
        state_indices_p[num_prefills:] = PAD_SLOT_ID
        num_decode_rows = static.state_indices_d.shape[0]
        if state_indices_tensor_d is None:
            state_indices_tensor_d = self.state_indices_tensor_d[:0]
        state_indices_d, replayssm_state_indices_d = self._stage_static_decode_indices(
            metadata, state_indices_tensor_d[: metadata.num_decodes], num_decode_rows
        )
        return replace(
            static,
            state_indices_p=state_indices_p,
            state_indices_d=state_indices_d,
            replayssm_state_indices_d=replayssm_state_indices_d,
        )

    def update_block_table(
        self,
        metadata: Mamba2AttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Mamba2AttentionMetadata:
        new_metadata = super().update_block_table(metadata, blk_table, slot_mapping)
        if metadata.static is not None:
            new_metadata = replace(
                new_metadata,
                static=self._stage_static_state_indices(
                    metadata.static,
                    new_metadata,
                    new_metadata.state_indices_tensor_d,
                    new_metadata.state_indices_tensor_p,
                ),
            )
        if metadata.checkpoint_meta is None:
            return new_metadata
        # Checkpoint destinations are block-table entries, so each group
        # re-gathers its own rather than writing into the source group's.
        return replace(
            new_metadata,
            checkpoint_meta=metadata.checkpoint_meta.regather_state_indices(blk_table),
        )
