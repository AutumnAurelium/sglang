import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AfmoeConfig, AfmoeForCausalLM

from sglang import Engine
from sglang.srt.utils.network import get_free_port
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="extra-a", runner_config="2-gpu-large")


class TestAfmoeDeepEP(CustomTestCase):
    def check_shared_experts(self, backend):
        torch.manual_seed(42)
        config = AfmoeConfig(
            vocab_size=256,
            hidden_size=2048,
            head_dim=128,
            num_attention_heads=16,
            num_key_value_heads=2,
            moe_intermediate_size=256,
            num_hidden_layers=1,
            num_dense_layers=0,
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=1,
            layer_types=["full_attention"],
            max_position_embeddings=128,
            attn_implementation="eager",
        )
        reference = AfmoeForCausalLM(config).eval().to(torch.bfloat16)
        ids = torch.randint(1, config.vocab_size, (1, 32))
        with torch.inference_mode():
            # Isolate the shared expert's contribution to the FFN.
            reference.model.layers[0].mlp.experts.down_proj.zero_()
            logits = reference(ids, use_cache=False).logits[0, :-1].float()
            expected = logits.log_softmax(-1).gather(1, ids[0, 1:, None]).flatten()
        with tempfile.TemporaryDirectory() as checkpoint:
            reference.save_pretrained(checkpoint, save_original_format=False)
            engine = Engine(
                model_path=checkpoint,
                port=get_free_port(),
                dtype="bfloat16",
                skip_tokenizer_init=True,
                tp_size=2,
                ep_size=2,
                attn_dp_size=2,
                moe_a2a_backend=backend,
                moe_runner_backend="deep_gemm",
                deepep_mode="normal",
                disable_shared_experts_fusion=True,
                disable_cuda_graph=True,
                disable_radix_cache=True,
                chunked_prefill_size=128,
                max_prefill_tokens=128,
                max_total_tokens=2048,
            )
            try:
                for rank in (0, 1):
                    with self.subTest(rank=rank):
                        result = engine.generate(
                            input_ids=ids[0].tolist(),
                            sampling_params={"max_new_tokens": 0},
                            routed_dp_rank=rank,
                            return_logprob=True,
                            logprob_start_len=0,
                        )
                        self.assertEqual(result["meta_info"]["dp_rank"], rank)
                        rows = result["meta_info"]["input_token_logprobs"][1:]
                        actual = torch.tensor([row[0] for row in rows])
                        torch.testing.assert_close(actual, expected, atol=0.05, rtol=0)
            finally:
                engine.shutdown()

    def test_deepep(self):
        self.check_shared_experts("deepep")

    @patch.dict(
        os.environ, EP_DISABLE_GIN="1", EP_REUSE_NCCL_COMM="0", NCCL_CUMEM_ENABLE="1"
    )
    def test_deepep_v2(self):
        self.check_shared_experts("deepep_v2")


if __name__ == "__main__":
    unittest.main()
