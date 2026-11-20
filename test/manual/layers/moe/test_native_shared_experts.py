import unittest
from unittest.mock import patch

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe import topk
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase


class TestNativeSharedExperts(CustomTestCase):
    def test_shared_expert_weights(self):
        logits = torch.tensor([[0.375, 0.125, 0.0625]]).logit()
        hidden = torch.ones(1, 2)
        with (
            get_context().override_server_args(),
            patch.object(topk, "fused_topk_native", topk.fused_topk_torch_native),
        ):
            for fold_scale in (True, False):
                with self.subTest(fold_scale=fold_scale):
                    config = topk.TopKConfig(
                        top_k=4,
                        scoring_func="sigmoid",
                        torch_native=True,
                        shared_experts=SharedExpertsConfig(num_fused_shared_experts=2),
                        routed_scaling_factor=2.0,
                        apply_routed_scaling_factor_on_output=fold_scale,
                    )
                    output = topk.select_experts(hidden, logits, config)
                    ids = output.topk_ids[0]
                    weights = output.topk_weights[0]
                    self.assertEqual(ids[2:].tolist(), [3, 4])
                    routed_ids, order = ids[:2].sort()
                    self.assertEqual(routed_ids.tolist(), [0, 1])
                    # Either the router or the expert epilogue applies the scale.
                    effective = weights if fold_scale else weights * 2
                    torch.testing.assert_close(
                        effective[:2][order], torch.tensor([1.5, 0.5])
                    )
                    torch.testing.assert_close(effective[2:], torch.ones(2))


if __name__ == "__main__":
    unittest.main()
