import unittest
import torch
from run_recursive_model import build_graph
from scrubbing import Scrubber


class AdapterTests(unittest.TestCase):
    def test_tensor_categories_are_not_matched_by_object_identity(self):
        from scrubbing import exact_key
        with self.assertRaisesRegex(TypeError, "categorical encoder"):
            exact_key(torch.tensor(1.0))

    def setUp(self):
        self.data = [0, 0, 1, 1]
        def prefix(i):
            return torch.tensor([[[float(self.data[i])]]])
        prefix.dtype = torch.float32
        self.prefix = prefix

    def hypothesis(self, scale):
        return build_graph(self.prefix, lambda x: x, lambda x: float(x.item()),
                           lambda x: scale * x, lambda x: scale * x, self.data)

    def test_gain_decomposition_preserves_original_graph(self):
        for scale in (0.0, 0.3, 1.0, 2.0, 10.0):
            h = self.hypothesis(scale)
            for i, expected in enumerate(self.data):
                self.assertEqual(h.G.evaluate(i), 0)
                self.assertAlmostEqual(h.G.evaluate(i, "prediction"), expected, places=6)

    def test_full_state_control_and_tree_replay(self):
        s = Scrubber(self.hypothesis(1), range(4))
        for i in range(4):
            for seed in range(10):
                value, trace = s.run(i, seed)
                self.assertEqual(value, 0)
                self.assertEqual(s.replay_tree(trace), value)

    def test_empty_support_exposes_false_invariance(self):
        s = Scrubber(self.hypothesis(0), range(4))
        values = [s.run(i, seed)[0] for i in range(4) for seed in range(30)]
        self.assertGreater(sum(values), 20)
        self.assertLess(sum(values), 100)

    def test_continuous_metrics_handle_silence(self):
        from run_baseline import metrics
        r = metrics(torch.tensor([0., 1.]), torch.tensor([0., 0.]), 0.)
        self.assertIsNone(r["correlation"])
        self.assertEqual(r["false_firing"], 0.5)
        self.assertTrue(torch.isfinite(torch.tensor(r["nrmse"])))


if __name__ == "__main__":
    unittest.main()
