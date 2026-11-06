"""Semantic checks for the reference algorithm, runnable without model weights."""
import itertools
import random
import unittest

from scrubbing import Graph, Hypothesis, Node, Scrubber


def arithmetic(correct=True):
    g = Graph([Node("x"), Node("a", ("x",), lambda x: x[0]),
               Node("b", ("x",), lambda x: x[1]),
               Node("noise", ("x",), lambda x: x[2]),
               Node("out", ("a", "b", "noise"), lambda a, b, n: a + 2 * b)], "x", "out")
    i = Graph([Node("x"), Node("a", ("x",), lambda x: x[0] if correct else x[1]),
               Node("b", ("x",), lambda x: x[1]),
               Node("out", ("a", "b"), lambda a, b: a + 2 * b)], "x", "out")
    return Hypothesis(g, i, {n: n for n in i.nodes}), list(itertools.product(range(2), repeat=3))


def shared_ancestor():
    g = Graph([Node("x"), Node("a", ("x",), lambda x: x),
               Node("b", ("a",), lambda a: a), Node("c", ("a",), lambda a: a),
               Node("out", ("b", "c"), lambda b, c: b - c)], "x", "out")
    i = Graph([Node("x"), Node("a", ("x",), lambda x: 0),
               Node("b", ("a",), lambda a: 0), Node("c", ("a",), lambda a: 0),
               Node("out", ("b", "c"), lambda b, c: 0)], "x", "out")
    return Hypothesis(g, i, {n: n for n in i.nodes}), [0, 1]


class ScrubbingTests(unittest.TestCase):
    def test_correct_hypothesis_preserves_each_reference(self):
        h, data = arithmetic()
        s = Scrubber(h, data)
        for ref, x in enumerate(data):
            for seed in range(20):
                value, trace = s.run(ref, seed)
                self.assertEqual(value, h.G.evaluate(x))
                self.assertEqual(s.replay_tree(trace), value)

    def test_false_feature_meaning_changes_output(self):
        h, data = arithmetic(False)
        s = Scrubber(h, data)
        dist = s.enumerate_distribution(0)
        self.assertAlmostEqual(dist[0], 0.5)
        self.assertAlmostEqual(dist[1], 0.5)

    def test_shared_ancestor_is_sampled_separately_per_path(self):
        h, data = shared_ancestor()
        s = Scrubber(h, data)
        expected = {-1: 0.25, 0: 0.5, 1: 0.25}
        self.assertEqual(s.enumerate_distribution(0), expected)
        seen = set()
        for seed in range(100):
            value, trace = s.run(0, seed)
            seen.add(value)
            self.assertEqual(s.replay_tree(trace), value)
            paths = [r["path"] for r in trace.calls if r["node"] == "a"]
            self.assertEqual(len(paths), 2)
            self.assertNotEqual(paths[0], paths[1])
        self.assertEqual(seen, {-1, 0, 1})

    def test_unimportant_parents_share_a_donor(self):
        g = Graph([Node("x"), Node("p", ("x",), lambda x: x),
                   Node("q", ("x",), lambda x: x),
                   Node("out", ("x", "p", "q"), lambda x, p, q: p - q)], "x", "out")
        i = Graph([Node("x"), Node("out", ("x",), lambda x: 0)], "x", "out")
        s = Scrubber(Hypothesis(g, i, {"x": "x", "out": "out"}), [0, 1])
        self.assertEqual(s.enumerate_distribution(0), {0: 1.0})
        for seed in range(20):
            value, trace = s.run(0, seed)
            self.assertEqual(value, 0)
            self.assertEqual(len({r["donor"] for r in trace.unimportant}), 1)
            self.assertEqual(s.replay_tree(trace), value)

    def test_empirical_duplicate_rows_keep_probability_mass(self):
        h, _ = shared_ancestor()
        s = Scrubber(h, [0, 0, 1])
        self.assertEqual(s.agreeing("a", 0), [0, 1, 2])
        dist = s.enumerate_distribution(0)
        self.assertAlmostEqual(dist[-1], 2 / 9)
        self.assertAlmostEqual(dist[1], 2 / 9)
        self.assertAlmostEqual(dist[0], 5 / 9)

    def test_exact_matching_and_singletons_are_reported(self):
        g = Graph([Node("x"), Node("out", ("x",), lambda x: x)], "x", "out")
        s = Scrubber(Hypothesis(g, g, {"x": "x", "out": "out"}), [0.1, 0.10000001])
        value, trace = s.run(0, 7)
        self.assertEqual(value, 0.1)
        self.assertEqual(trace.summary()["singleton_fraction"], 1.0)
        self.assertEqual(trace.summary()["self_fraction"], 1.0)

    def test_reproducibility_and_external_rng(self):
        h, data = shared_ancestor()
        s = Scrubber(h, data)
        a, ta = s.run(seed=27)
        b, tb = s.run(rng=random.Random(27))
        self.assertEqual((a, ta), (b, tb))

    def test_signature_and_graph_validation(self):
        with self.assertRaisesRegex(ValueError, "signature"):
            Graph([Node("x"), Node("out", ("x",), lambda a, b: 0)], "x", "out")
        with self.assertRaisesRegex(ValueError, "cycle"):
            Graph([Node("x"), Node("out", ("x", "out"), lambda x, y: 0)], "x", "out")
        with self.assertRaisesRegex(ValueError, "missing parent"):
            Graph([Node("x"), Node("out", ("absent",), lambda x: x)], "x", "out")

    def test_correspondence_validation(self):
        h, _ = arithmetic()
        c = dict(h.correspondence)
        c["a"] = "noise"
        Hypothesis(h.G, h.I, c)  # valid graph homomorphism, even if semantically false
        c["b"] = "noise"
        with self.assertRaisesRegex(ValueError, "injective"):
            Hypothesis(h.G, h.I, c)
        c = dict(h.correspondence)
        c["out"] = "noise"
        with self.assertRaisesRegex(ValueError, "input/output"):
            Hypothesis(h.G, h.I, c)
        with self.assertRaisesRegex(ValueError, "cover"):
            Hypothesis(h.G, h.I, {})

    def test_empty_dataset_and_invalid_reference(self):
        h, data = arithmetic()
        with self.assertRaisesRegex(ValueError, "empty"):
            Scrubber(h, [])
        with self.assertRaises(IndexError):
            Scrubber(h, data).run(-1)

    def test_nan_classes_are_rejected(self):
        g = Graph([Node("x"), Node("out", ("x",), lambda x: x)], "x", "out")
        s = Scrubber(Hypothesis(g, g, {"x": "x", "out": "out"}), [float("nan")])
        with self.assertRaisesRegex(ValueError, "NaN"):
            s.run(0)


if __name__ == "__main__":
    unittest.main()
