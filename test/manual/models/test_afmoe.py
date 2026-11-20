import unittest

from sglang.srt.configs.afmoe import AfmoeConfig
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import CustomTestCase


class TestAfmoeConfig(CustomTestCase):
    def test_derived_attention_dimensions(self):
        config = AfmoeConfig(hidden_size=384, num_attention_heads=4)
        self.assertEqual(config.head_dim, 96)
        self.assertEqual(config.num_key_value_heads, 4)

    def test_legacy_routing_fields(self):
        config = AfmoeConfig.from_dict(
            {
                "score_func": "softmax",
                "route_norm": False,
                "n_group": 4,
                "topk_group": 2,
            }
        )
        self.assertEqual(config.score_func, "softmax")
        self.assertFalse(config.route_norm)
        self.assertEqual((config.n_group, config.topk_group), (4, 2))


class TestTrinityMini(GSM8KMixin, DefaultServerBase):
    model = "arcee-ai/Trinity-Mini"
    other_args = ["--context-length", "4096"]
    # Pre-port BF16 baseline: 180/200 on the default five-shot fixture.
    gsm8k_score_threshold = 0.85


if __name__ == "__main__":
    unittest.main()
