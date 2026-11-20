import tempfile
import unittest
from pathlib import Path
from types import MethodType
from unittest.mock import Mock

import torch
from transformers import AfmoeConfig, AfmoeForCausalLM, FineGrainedFP8Config

from sglang import Engine
from sglang.srt.configs.afmoe import AfmoeConfig as SGLangAfmoeConfig
from sglang.srt.models.afmoe import AfmoeForCausalLM as SGLangAfmoeForCausalLM
from sglang.srt.models.afmoe import AfmoeMoE
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-large")


class TestAfmoeConfig(CustomTestCase):
    def test_experts_reject_unsupported_activation(self):
        config = SGLangAfmoeConfig(
            num_dense_layers=0, num_shared_experts=0, hidden_act="gelu"
        )
        with self.assertRaisesRegex(ValueError, "SiLU"):
            AfmoeMoE(config, layer_id=0)

    def test_load_calibrated_kv_scales(self):
        model = Mock(spec=SGLangAfmoeForCausalLM)
        model.start_layer, model.end_layer = 0, 1
        model.config = SGLangAfmoeConfig(tie_word_embeddings=False)
        model.num_fused_shared_experts = 0
        model._load_packed_experts.return_value = False
        prefix = "model.layers.0.self_attn"
        scales = {
            f"{prefix}.attn.{kind}_scale": torch.tensor(-1.0) for kind in ("k", "v")
        }
        model.named_parameters.return_value = scales.items()
        SGLangAfmoeForCausalLM.load_weights(
            model,
            [
                (f"{prefix}.k_proj.k_scale", torch.tensor(0.125)),
                (f"{prefix}.v_proj.v_scale", torch.tensor(0.25)),
            ],
        )
        self.assertEqual(scales[f"{prefix}.attn.k_scale"].item(), 0.125)
        self.assertEqual(scales[f"{prefix}.attn.v_scale"].item(), 0.25)

    def test_legacy_defaults_and_overrides(self):
        config = SGLangAfmoeConfig(
            hidden_size=384,
            num_attention_heads=4,
            num_hidden_layers=2,
            global_attn_every_n_layers=2,
        )
        self.assertEqual(config.head_dim, 96)
        self.assertEqual(config.num_key_value_heads, 4)
        self.assertEqual(config.layer_types, ["sliding_attention", "full_attention"])
        self.assertEqual(
            (config.score_func, config.route_norm, config.n_group, config.topk_group),
            ("sigmoid", True, 1, 1),
        )
        config = SGLangAfmoeConfig.from_dict(
            dict(
                head_dim=64,
                score_func="softmax",
                route_norm=False,
                n_group=4,
                topk_group=2,
            )
        )
        self.assertEqual(
            (
                config.head_dim,
                config.score_func,
                config.route_norm,
                config.n_group,
                config.topk_group,
            ),
            (64, "softmax", False, 4, 2),
        )


class AfmoeParityTestCase(CustomTestCase):
    num_experts = 8

    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        config = AfmoeConfig(
            vocab_size=256,
            hidden_size=256,
            intermediate_size=384,
            moe_intermediate_size=128,
            num_hidden_layers=3,
            num_dense_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_experts=self.num_experts,
            num_experts_per_tok=2,
            num_shared_experts=1,
            route_scale=2.826,
            global_attn_every_n_layers=2,
            sliding_window=32,
            max_position_embeddings=128,
            mup_enabled=True,
            attn_implementation="eager",
        )
        reference = AfmoeForCausalLM(config).eval().to(torch.bfloat16)
        ids = torch.randint(1, config.vocab_size - 1, (1, 65))
        with torch.inference_mode():
            # Give each token a preferred pair with a margin. Near ties can
            # otherwise flip the discrete route on BF16 or FP8 rounding alone.
            pairs = torch.arange(config.num_experts // 2)
            channels = pairs * (config.hidden_size // len(pairs))
            tokens = torch.arange(config.vocab_size)
            reference.model.embed_tokens.weight[
                tokens, channels[tokens % len(pairs)]
            ] = 1.5
            for layer in reference.model.layers[config.num_dense_layers :]:
                gate = layer.mlp.router.gate.weight
                gate.normal_(std=0.002)
                gate[2 * pairs, channels] = 0.3
                gate[2 * pairs + 1, channels] = 0.15
                layer.mlp.expert_bias.copy_(
                    torch.linspace(-0.04, 0.04, config.num_experts)
                )
        self.reference = reference
        self.ids = ids
        self.checkpoint = self.enterContext(tempfile.TemporaryDirectory())
        reference.save_pretrained(self.checkpoint, save_original_format=False)

    @torch.inference_mode()
    def reference_logprobs(self, ids=None):
        ids = self.ids if ids is None else ids
        logits = self.reference(ids, use_cache=False).logits[0, :-1].float()
        return logits.log_softmax(-1).gather(1, ids[0, 1:, None]).flatten()

    def check_engine(self, *, lora_path=None, atol=0.05, decode_tokens=0, **options):
        expected = self.reference_logprobs()
        engine = Engine(
            model_path=self.checkpoint,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_total_tokens=2048,
            disable_cuda_graph=not decode_tokens,
            moe_runner_backend="triton",
            **options,
        )
        try:
            response = engine.generate(
                input_ids=self.ids[0].tolist(),
                sampling_params={
                    "max_new_tokens": decode_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                },
                return_logprob=True,
                logprob_start_len=0,
                lora_path=lora_path,
            )
        finally:
            engine.shutdown()
        rows = response["meta_info"]["input_token_logprobs"][1:]
        self.assertEqual([row[1] for row in rows], self.ids[0, 1:].tolist())
        actual = torch.tensor([row[0] for row in rows])
        torch.testing.assert_close(actual, expected, rtol=0, atol=atol)
        if decode_tokens:
            generated = response["output_ids"]
            self.assertEqual(len(generated), decode_tokens)
            ids = torch.tensor([self.ids[0].tolist() + generated])
            expected = self.reference_logprobs(ids)[-decode_tokens:]
            actual = torch.tensor(
                [row[0] for row in response["meta_info"]["output_token_logprobs"]]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=atol)


class TestAfmoe(AfmoeParityTestCase):
    def test_packed_checkpoint_matches_transformers(self):
        with torch.inference_mode():
            output = self.reference(
                self.ids, use_cache=False, output_router_logits=True
            )
            for logits, layer in zip(
                output.router_logits,
                self.reference.model.layers[self.reference.config.num_dense_layers :],
                strict=True,
            ):
                scores = logits.float().sigmoid()
                ids = (scores + layer.mlp.expert_bias).topk(2).indices
                self.assertEqual(ids.unique().numel(), self.num_experts)
                self.assertGreater(ids.sort(dim=-1).values.unique(dim=0).shape[0], 1)
                self.assertTrue(
                    (scores.gather(-1, ids).diff(dim=-1).abs() > 0.01).any()
                )
        self.check_engine()

    def test_compiled_single_token_decode(self):
        self.check_engine(
            decode_tokens=3,
            enable_torch_compile=True,
            cuda_graph_bs_decode=[1],
            disable_prefill_cuda_graph=True,
        )

    def test_fp8_kv_cache(self):
        # Chunked prefill forces attention to read previously cached keys and values.
        for dtype, backend in (("fp8_e4m3", "fa3"), ("fp8_e5m2", "triton")):
            with self.subTest(dtype=dtype):
                self.check_engine(
                    kv_cache_dtype=dtype,
                    attention_backend=backend,
                    chunked_prefill_size=32,
                    atol=0.10,
                )

    def test_attention_lora(self):
        from peft import LoraConfig, get_peft_model

        base_logprobs = self.reference_logprobs()
        adapter = Path(self.checkpoint) / "adapter"
        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj", "self_attn.gate_proj"],
            init_lora_weights=False,
        )
        self.reference = get_peft_model(self.reference, config).eval()
        self.reference.save_pretrained(adapter)
        self.assertGreater(
            (self.reference_logprobs() - base_logprobs).abs().max().item(), 0.1
        )
        self.check_engine(
            enable_lora=True, lora_paths={"adapter": str(adapter)}, lora_path="adapter"
        )

    def test_legacy_softmax_and_split_experts(self):
        # The released Trinity router supports softmax. The native HF model
        # always uses sigmoid, so supply the legacy equation for the oracle.
        def forward(router, hidden_states, expert_bias):
            logits = router.gate(hidden_states.flatten(0, 1)).float()
            probs = logits.softmax(-1)
            ids = (probs + expert_bias).topk(router.top_k).indices
            return logits, probs.gather(1, ids) * router.route_scale, ids

        config = self.reference.config
        config.score_func = "softmax"
        for layer in self.reference.model.layers[config.num_dense_layers :]:
            layer.mlp.router.forward = MethodType(forward, layer.mlp.router)
        self.reference.save_pretrained(self.checkpoint, save_original_format=True)
        self.check_engine()


class TestAfmoeFP8(AfmoeParityTestCase):
    # Transformers needs one complete block across the router's output dimension.
    num_experts = 128

    def test_packed_fp8_experts(self):
        config = FineGrainedFP8Config(modules_to_not_convert=["lm_head"])
        quantized = AfmoeForCausalLM.from_pretrained(
            self.checkpoint,
            quantization_config=config,
            device_map="cuda",
            dtype=torch.bfloat16,
        )
        scales = quantized.model.layers[1].mlp.experts.gate_up_proj_scale_inv
        self.assertTrue(torch.all((scales > 0) & (scales < 1)))
        gate = quantized.model.layers[1].mlp.router.gate
        self.assertEqual(gate.weight.dtype, torch.float8_e4m3fn)
        self.assertTrue(
            torch.all((gate.weight_scale_inv > 0) & (gate.weight_scale_inv < 1))
        )
        for layer in quantized.model.layers:
            layer.self_attn.k_proj.register_buffer("k_scale", torch.tensor(0.125))
            layer.self_attn.v_proj.register_buffer("v_scale", torch.tensor(0.25))
        quantized.save_pretrained(self.checkpoint, save_original_format=False)
        # Reconstruct the reference from the actual FP8 blocks. This isolates
        # scale loading from the loss of precision when the checkpoint is made.
        weights = quantized.state_dict()
        with torch.no_grad():
            for name, param in self.reference.named_parameters():
                weight = weights[name]
                if weight.dtype == torch.float8_e4m3fn:
                    scales = weights[name + "_scale_inv"]
                    block_m, block_n = config.weight_block_size
                    scales = scales.repeat_interleave(block_m, -2).repeat_interleave(
                        block_n, -1
                    )
                    param.copy_(weight.float() * scales)
        del weights, quantized
        self.check_engine(
            fp8_gemm_runner_backend="triton",
            kv_cache_dtype="fp8_e4m3",
            attention_backend="fa3",
            chunked_prefill_size=32,
            atol=0.10,
        )


if __name__ == "__main__":
    unittest.main()
