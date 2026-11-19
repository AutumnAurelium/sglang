# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only AfMoE model compatible with HuggingFace weights.

AfMoE is a Mixture-of-Experts model with:
- Gated attention with sigmoid gating
- Q/K normalization with RMSNorm
- Dual normalization (pre/post for both attention and MLP)
- Sliding window attention for local layers
- muP (maximal update parameterization) scaling support
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
    is_dense_ffn_fully_dp,
)
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    LinearParallelGroup,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    adds_replicated_output_to_partial,
    filter_moe_weight_param_global_expert,
    get_num_fused_shared_experts,
    has_per_rank_fused_shared_slots,
    is_deepep_class_backend,
    is_moe_input_scattered_across_dp_ranks,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.utils.layer_list import make_pp_layers
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_exec, get_lora, get_model, get_parallel
from sglang.srt.utils import LazyValue, is_cpu, is_cuda


def get_attention_sliding_window_size(config: PretrainedConfig) -> Optional[int]:
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        return None
    if sliding_window <= 0:
        return None
    # Align with other local attention implementations (see gpt_oss).
    return sliding_window - 1


def is_moe_layer(config: PretrainedConfig, layer_id: int) -> bool:
    if not config.num_experts or not 0 <= layer_id < config.num_hidden_layers:
        return False
    if hasattr(config, "num_dense_layers"):
        return layer_id >= config.num_dense_layers
    if hasattr(config, "first_k_dense_replace") and hasattr(config, "moe_layer_freq"):
        base = config.first_k_dense_replace
        return layer_id >= base and (layer_id - base) % config.moe_layer_freq == 0
    return False


class AfmoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        parallel_group: LinearParallelGroup = "tp",
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            parallel_group=parallel_group,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            parallel_group=parallel_group,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class AfmoeMoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        num_experts = config.num_experts
        if config.score_func != "sigmoid":
            raise ValueError(
                f"AFMoE requires sigmoid routing, got {config.score_func!r}"
            )
        num_fused = get_num_fused_shared_experts()
        self.num_fused_shared_experts = num_fused
        shared_slots = (
            num_fused * get_parallel().moe_ep_size
            if has_per_rank_fused_shared_slots(num_fused)
            else num_fused
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.expert_bias = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32), requires_grad=False
        )

        runner = get_moe_runner_backend()
        if runner.is_flashinfer_mxfp4() and not getattr(
            quant_config, "is_fp4_experts", False
        ):
            raise ValueError(
                "AFMoE requires a MoE runner that consumes precomputed routing weights."
            )
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=num_experts
            + shared_slots
            + get_exec().moe.ep_num_redundant_experts,
            num_fused_shared_experts=num_fused,
            top_k=config.num_experts_per_tok + num_fused,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.TopK,
        )
        # The router scales only the routed weights. A fused shared slot must
        # contribute exactly once, just like the separate shared MLP.
        self.route_scale = config.route_scale
        self.scale_after_experts = is_cpu()
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            scoring_func="sigmoid",
            renormalize=config.route_norm,
            use_grouped_topk=(config.n_group or 1) > 1,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            correction_bias=self.expert_bias,
            routed_scaling_factor=self.route_scale,
            apply_routed_scaling_factor_on_output=not self.scale_after_experts,
            is_fp4_experts=getattr(quant_config, "is_fp4_experts", False),
        )
        self.topk.bind_runner_backend(
            self.experts.runner_backend,
            routing_method_type=self.experts.routing_method_type,
        )

        self.shared_experts = None
        self.shared_replicated = is_moe_input_scattered_across_dp_ranks()
        if config.num_shared_experts and not num_fused:
            self.shared_experts = AfmoeMLP(
                config.hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
                config.hidden_act,
                quant_config=quant_config,
                parallel_group="replicated" if self.shared_replicated else "tp",
                prefix=add_prefix("shared_experts", prefix),
            )

    def get_moe_weights(self):
        # EPLB may move routed experts, but never the fixed shared slots.
        num_routed = self.experts.num_local_experts - self.num_fused_shared_experts
        return [
            param.data[:num_routed]
            for name, param in self.experts.named_parameters()
            if name != "correction_bias"
            and filter_moe_weight_param_global_expert(
                name, param, self.experts.num_local_experts
            )
        ]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_output = None
        if hidden_states.shape[0]:
            if self.shared_experts is not None and (
                not self.shared_replicated or adds_replicated_output_to_partial()
            ):
                shared_output = self.shared_experts(hidden_states)
            logits, _ = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, logits)
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        output = self.experts(hidden_states, topk_output)
        if self.scale_after_experts:
            output = output * self.route_scale
        if shared_output is not None:
            output = output + shared_output
        return output


class AfmoeAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_parallel().attn_tp_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(config, "head_dim", hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        rope_theta = config.rope_parameters["rope_theta"]
        rope_scaling = config.rope_parameters
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        layer_types = getattr(config, "layer_types", None)
        self.is_local_attention = (
            layer_types is not None and layer_types[layer_id] == "sliding_attention"
        )
        sliding_window = (
            get_attention_sliding_window_size(config) if self.is_local_attention else -1
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            parallel_group="attn_tp",
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            reduce_results=False,
            parallel_group="attn_tp",
            prefix=add_prefix("o_proj", prefix),
        )
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            parallel_group="attn_tp",
            prefix=add_prefix("gate_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        eps = getattr(config, "rms_norm_eps", 1e-5)
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        self.sliding_window = sliding_window

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
        )

        if self.is_local_attention:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)

        gate_vals, _ = self.gate_proj(hidden_states)
        attn_output = attn_output * torch.sigmoid(gate_vals)
        output, _ = self.o_proj(attn_output)
        return output


class AfmoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        self.self_attn = AfmoeAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        if is_moe_layer(config, layer_id):
            self.mlp = AfmoeMoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = AfmoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                parallel_group="replicated" if is_dense_ffn_fully_dp() else "tp",
            )

        eps = getattr(config, "rms_norm_eps", 1e-5)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=eps)

        attn, ffn = self.stage_facts(
            config, layer_id, self.post_attention_layernorm, self.post_mlp_layernorm
        )
        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm), (ffn, self.pre_mlp_layernorm)
        )

    @staticmethod
    def stage_facts(config, layer_id, post_attn_norm=None, post_mlp_norm=None):
        # AFMoE normalizes each completed branch before the residual addition.
        return (
            declare_attn(output_transform=OutputTransform(post_attn_norm)),
            declare_ffn(
                sparse=is_moe_layer(config, layer_id),
                next_layer_sparse=is_moe_layer(config, layer_id + 1),
                output_transform=OutputTransform(post_mlp_norm),
            ),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.mlp(hidden_states)
        return self.ffn_boundary.finish(hidden_states, forward_batch)


class AfmoeModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.pp_group = get_parallel().pp_group
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: AfmoeDecoderLayer(
                config, idx, quant_config=quant_config, prefix=prefix
            ),
            prefix=add_prefix("layers", prefix),
            stage_facts=lambda idx: AfmoeDecoderLayer.stage_facts(config, idx),
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.pp_group.is_last_rank
            else PPMissingLayer()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor | PPProxyTensors:
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            if getattr(self.config, "mup_enabled", False):
                hidden_states = hidden_states * (self.config.hidden_size**0.5)
            residual_batch.start(forward_batch)
        else:
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )

        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden_states = self.layers[i](positions, hidden_states, forward_batch)
        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)
        return residual_batch.final_norm(
            hidden_states, forward_batch, self.norm, skip_empty=True
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


class AfmoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if get_lora().enable_lora or get_lora().lora_paths:
            raise ValueError("AFMoE LoRA cannot map the separate attention gate.")
        if get_model().kv_cache_dtype in ("fp8_e4m3", "fp8_e5m2", "mxfp8"):
            raise ValueError("AFMoE does not load FP8 KV-cache scales.")
        self.config = config
        self.quant_config = quant_config
        self.model = AfmoeModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.num_fused_shared_experts = get_num_fused_shared_experts()
        self.pp_group = self.model.pp_group
        if self.pp_group.world_size == 1 and config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        elif self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_parallel().enable_dp_lm_head,
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)
        self.routed_experts_weights_of_layer = LazyValue(
            lambda: {
                i: self.model.layers[i].mlp.get_moe_weights()
                for i in range(self.start_layer, self.end_layer)
                if is_moe_layer(config, i)
            }
        )

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=config.n_group,
        )

    @staticmethod
    def num_fusable_shared_experts(hf_config):
        return hf_config.num_shared_experts or 0

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        if hf_config.num_shared_experts != 1:
            return "AFMoE shared expert fusion requires one shared expert."
        if not is_cuda() or torch.cuda.get_device_capability() < (8, 0):
            return "AFMoE shared expert fusion requires SM80 or newer."
        runner = get_moe_runner_backend()
        if quant_config is None and (
            runner.is_flashinfer_trtllm() or runner.is_flashinfer_trtllm_routed()
        ):
            return "The BF16 TRT-LLM runner does not support fused shared experts."
        if quant_config is not None and not (
            hasattr(quant_config, "can_fuse_shared_expert")
            and quant_config.can_fuse_shared_expert()
        ):
            return "The shared and routed experts require compatible quantization."
        if not get_moe_a2a_backend().is_none():
            if not (
                get_exec().moe.enforce_shared_experts_fusion
                and is_deepep_class_backend()
            ):
                return "AFMoE all-to-all fusion requires enforced DeepEP shared slots."
        elif get_parallel().moe_ep_size > 1:
            return "AFMoE shared expert fusion under EP requires all-to-all dispatch."
        return None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor | PPProxyTensors:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return get_attention_sliding_window_size(self.config)

    def _unpack_expert_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Transformers stores all routed experts in two 3-D parameters. Feed
        # their views through the same TP/EP loader as per-expert checkpoints.
        for name, weight in weights:
            prefix, _, projection = name.removesuffix(".weight").rpartition(".")
            if not prefix.endswith(".mlp.experts") or projection not in (
                "gate_up_proj",
                "down_proj",
            ):
                yield name, weight
                continue
            shards = (
                ("gate_proj", "up_proj")
                if projection == "gate_up_proj"
                else ("down_proj",)
            )
            if (
                weight.ndim != 3
                or weight.shape[0] != self.config.num_experts
                or weight.shape[1] % len(shards)
            ):
                raise ValueError(f"Invalid packed expert tensor {name}: {weight.shape}")
            for shard, expert_weights in zip(shards, weight.chunk(len(shards), dim=1)):
                for expert_id, expert_weight in enumerate(expert_weights.unbind()):
                    yield f"{prefix}.{expert_id}.{shard}.weight", expert_weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        stacked_params_mapping = [
            (".self_attn.qkv_proj.", ".self_attn.q_proj.", "q"),
            (".self_attn.qkv_proj.", ".self_attn.k_proj.", "k"),
            (".self_attn.qkv_proj.", ".self_attn.v_proj.", "v"),
            (".mlp.gate_up_proj.", ".mlp.gate_proj.", 0),
            (".mlp.gate_up_proj.", ".mlp.up_proj.", 1),
            (".shared_experts.gate_up_proj.", ".shared_experts.gate_proj.", 0),
            (".shared_experts.gate_up_proj.", ".shared_experts.up_proj.", 1),
        ]
        num_fused = self.num_fused_shared_experts
        expert_mapping = {
            entry[1]: entry
            for entry in FusedMoE.make_expert_params_mapping(
                "gate_proj",
                "down_proj",
                "up_proj",
                (self.config.num_experts or 0) + num_fused,
            )
        }
        expert_projection = re.compile(
            r"experts\.\d+\.(?:gate_proj|up_proj|down_proj)\."
        )
        params = dict(self.named_parameters())
        for name, weight in self._unpack_expert_weights(weights):
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and not self.start_layer <= layer_id < self.end_layer
            ):
                continue
            if self.config.tie_word_embeddings:
                if self.pp_group.is_first_rank and name == "lm_head.weight":
                    name = "model.embed_tokens.weight"
                elif (
                    self.pp_group.is_last_rank
                    and self.pp_group.world_size > 1
                    and name == "model.embed_tokens.weight"
                ):
                    name = "lm_head.weight"
            name = name.replace(".mlp.router.gate.", ".mlp.gate.")
            if num_fused:
                name = name.replace(
                    ".mlp.shared_experts.", f".mlp.experts.{self.config.num_experts}."
                )
            if match := expert_projection.search(name):
                param_name, weight_name, expert_id, shard_id = expert_mapping[
                    match.group()
                ]
                name = name.replace(weight_name, param_name)
                if name in params:
                    param = params[name]
                    param.weight_loader(
                        param, weight, name, expert_id=expert_id, shard_id=shard_id
                    )
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    param = params.get(name.replace(weight_name, param_name))
                    if param is not None:
                        param.weight_loader(param, weight, shard_id)
                    break
            else:
                if name in params:
                    param = params[name]
                    getattr(param, "weight_loader", default_weight_loader)(
                        param, weight
                    )


EntryClass = AfmoeForCausalLM
