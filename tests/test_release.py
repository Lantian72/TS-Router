"""Regression checks against retained outputs captured from the original 4x13 router."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ts_router import TSRouter, run_specialist, zscore


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = TSRouter(device="cpu")

    def test_preserves_all_four_heads_against_original(self):
        reference = json.loads((ROOT / "tests/pruning_reference.json").read_text())
        with torch.inference_mode():
            actual = self.model.router(torch.tensor(reference["features"], dtype=torch.float32))
        expected = torch.tensor(reference["original_retained_logits"], dtype=torch.float32)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(actual.shape, (len(reference["features"]), 4, 11))
        self.assertEqual(self.model.heads, reference["head_order"])
        self.assertEqual(self.model.experts, reference["retained_expert_order"])

    def test_all_eleven_specialists_produce_scores(self):
        rng = np.random.default_rng(12)
        t = np.arange(1024)
        series = np.sin(t * 2 * np.pi / 64) + rng.normal(0, 0.05, len(t))
        series[780:800] += 3
        train = zscore(series[:512, None])
        test = zscore(series[512:, None])
        for expert in self.model.experts:
            with self.subTest(expert=expert):
                scores = run_specialist(expert, train, test)
                self.assertEqual(scores.shape, (512,))
                self.assertTrue(np.isfinite(scores).all())

    def test_encoder_matches_original_checkpoint(self):
        import pandas as pd
        data = pd.read_csv(ROOT / "examples/synthetic_univariate.csv")["value"].to_numpy()
        actual = self.model.embed(data).cpu().numpy()
        expected = np.load(ROOT / "tests/encoder_reference.npy")
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_label_free_multivariate_prediction(self):
        import pandas as pd
        data = pd.read_csv(ROOT / "examples/synthetic_multivariate.csv").drop(columns="Label").values
        result = self.model.predict(data, train_index=512, head="all")
        self.assertEqual(set(result["scores"]), set(self.model.heads))
        for head, score in result["scores"].items():
            self.assertEqual(score.shape, (512,))
            self.assertTrue(np.isfinite(score).all())
            self.assertGreaterEqual(score.min(), 0)
            self.assertLessEqual(score.max(), 1)
            self.assertEqual(len(result["selected_experts"][head]), 2)
            for selected in result["selected_experts"][head]:
                self.assertEqual(len(selected), 3)
                self.assertFalse(set(selected) & {"KNN", "HBOS"})


if __name__ == "__main__":
    unittest.main()
