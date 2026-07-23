"""The mean-floor bisection objectives (free_mean_dense / free_mean_topk).

`prune_by_magnitude_bisection` bisects on whichever sufficiency `objective`
names. The two mean-floor objectives exist so the prune optimises the metric
actually REPORTED: pruning on free0 while quoting freeM_dense means the
bisection was never targeting the number in the table.

The contract each test pins:
  free              zero floor      -> site_means None, respect_topk False
  free_mean_dense   dense mean fill -> site_means passed, respect_topk False
  free_mean_topk    k-sparse fill   -> site_means passed, respect_topk True

and, load-bearing: the EMPTY-circuit floor must be measured under the SAME fill
as every phi(k), or the normalisation denominator belongs to a different metric
than the numerator.

`pin_values` is covered too: position-specific pins are ~671 MB per site, so a
caller sweeping several pruned variants of one circuit must be able to collect
them once and pass them in rather than have each call rebuild them.
"""
import pytest

from eval import magnitude_prune as mp
from store.circuits import Circuit, CircuitNode
from circuit.types.feature_id import FeatureID


def _circuit(n_members=10):
    c = Circuit(name="t")
    seed = c.add_node(CircuitNode(metadata={
        "feature_id": FeatureID(5, "mlp", 999), "role": "seed"}))
    for i in range(n_members):
        node = c.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(i % 5, "mlp", i),
            "role": "counterfactual_activator",
            "attribution_score": float(n_members - i),
        }))
        c.add_edge(node.uuid, seed.uuid, weight=float(n_members - i))
    return c


@pytest.fixture
def spy(monkeypatch):
    """Record how the eval was invoked for each call."""
    calls = []

    monkeypatch.setattr(mp, "measure_seed_activation", lambda *a, **k: 1.0)
    monkeypatch.setattr(mp, "upstream_sites", lambda *a, **k: {(0, "mlp"), (1, "mlp")})

    def fake_activation(inference, bank, keep_indices, in_scope, *a, **k):
        K = sum(len(v) for v in keep_indices.values())
        calls.append({"K": K,
                      "site_means": k.get("site_means"),
                      "respect_topk": k.get("respect_topk"),
                      "pin_values": k.get("pin_values")})
        return min(1.0, K / 6.0)

    monkeypatch.setattr(mp, "circuit_only_activation", fake_activation)

    import eval.floors as floors
    monkeypatch.setattr(floors, "collect_site_means",
                        lambda *a, **k: {(0, "mlp"): "MEANS", (1, "mlp"): "MEANS"})
    monkeypatch.setattr(floors, "collect_site_anchors",
                        lambda *a, **k: (None, {"pins": "COLLECTED"}))
    return calls


def _run(objective, **kw):
    return mp.prune_by_magnitude_bisection(
        inference=None, sae_bank=None, circuit=_circuit(),
        pos_tokens=None, seed_layer=5, seed_kind="mlp",
        seed_latent_idx=999, pos_argmax=None, objective=objective, **kw)


class TestObjectiveAccepted:
    @pytest.mark.parametrize("objective",
                             ["free", "free_mean_dense", "free_mean_topk", "pinned"])
    def test_valid_objectives_run(self, spy, objective):
        _run(objective)          # must not raise

    def test_invalid_objective_names_the_valid_set(self, spy):
        with pytest.raises(ValueError, match="free_mean_dense"):
            _run("freeM")


class TestFillRegime:
    def test_free_uses_the_zero_floor(self, spy):
        _run("free")
        assert all(c["site_means"] is None for c in spy)
        assert all(not c["respect_topk"] for c in spy)

    def test_free_mean_dense_passes_means_without_topk(self, spy):
        _run("free_mean_dense")
        assert all(c["site_means"] is not None for c in spy)
        assert all(not c["respect_topk"] for c in spy)

    def test_free_mean_topk_passes_means_with_topk(self, spy):
        _run("free_mean_topk")
        assert all(c["site_means"] is not None for c in spy)
        assert all(c["respect_topk"] for c in spy)

    def test_dense_and_topk_differ_only_in_the_fill(self, spy):
        _run("free_mean_dense")
        dense = [(c["K"], c["site_means"] is not None) for c in spy]
        spy.clear()
        _run("free_mean_topk")
        topk = [(c["K"], c["site_means"] is not None) for c in spy]
        assert dense == topk            # same bisection path, same floors


class TestFloorConsistency:
    """The empty-circuit floor must share the fill used for every phi(k)."""

    @pytest.mark.parametrize("objective", ["free", "free_mean_dense", "free_mean_topk"])
    def test_empty_floor_matches_the_measured_fill(self, spy, objective):
        _run(objective)
        empty = [c for c in spy if c["K"] == 0]
        assert empty, "the empty-circuit floor was never measured"
        for e in empty:
            for other in spy:
                assert e["respect_topk"] == other["respect_topk"]
                assert (e["site_means"] is None) == (other["site_means"] is None)


class TestPinValues:
    def test_pinned_collects_pins_when_not_supplied(self, spy):
        _run("pinned")
        assert any(c["pin_values"] is not None for c in spy)

    def test_supplied_pins_are_reused_not_recollected(self, spy, monkeypatch):
        import eval.floors as floors

        def explode(*a, **k):
            raise AssertionError("pins were re-collected despite being supplied")

        monkeypatch.setattr(floors, "collect_site_anchors", explode)
        _run("pinned", pin_values={"pins": "SUPPLIED"})
        assert all(c["pin_values"] == {"pins": "SUPPLIED"}
                   for c in spy if c["pin_values"] is not None)

    @pytest.mark.parametrize("objective", ["free", "free_mean_dense", "free_mean_topk"])
    def test_free_objectives_never_pin(self, spy, objective):
        """Pins would clamp kept latents to clean values — that is the DRIVERS
        question, not closure. Free objectives must ignore them even if passed."""
        _run(objective, pin_values={"pins": "SUPPLIED"})
        assert all(c["pin_values"] is None for c in spy)
