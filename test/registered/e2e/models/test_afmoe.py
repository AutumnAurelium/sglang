import tempfile
import unittest

import torch
from transformers import AfmoeConfig, AfmoeForCausalLM

from sglang import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


class TestAfmoe(CustomTestCase):
    def test_packed_checkpoint_matches_transformers(self):
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
            num_experts=8,
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
            for layer in reference.model.layers[config.num_dense_layers :]:
                layer.mlp.expert_bias.copy_(
                    torch.linspace(-0.2, 0.2, config.num_experts)
                )
            logits = reference(ids, use_cache=False).logits[0, :-1].float()
            expected = logits.log_softmax(-1).gather(1, ids[0, 1:, None]).flatten()

        with tempfile.TemporaryDirectory() as checkpoint:
            reference.save_pretrained(checkpoint, save_original_format=False)
            engine = Engine(
                model_path=checkpoint,
                skip_tokenizer_init=True,
                dtype="bfloat16",
                max_total_tokens=2048,
                disable_cuda_graph=True,
                moe_runner_backend="triton",
            )
            try:
                response = engine.generate(
                    input_ids=ids[0].tolist(),
                    sampling_params={"max_new_tokens": 0},
                    return_logprob=True,
                    logprob_start_len=0,
                )
            finally:
                engine.shutdown()
        rows = response["meta_info"]["input_token_logprobs"][1:]
        self.assertEqual([row[1] for row in rows], ids[0, 1:].tolist())
        actual = torch.tensor([row[0] for row in rows])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0.05)


if __name__ == "__main__":
    unittest.main()
