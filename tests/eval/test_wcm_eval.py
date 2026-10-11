"""Protocol-v1 scorer (eval/wcm_eval.py) and specificity (eval/wcm_specificity.py).

Part 1  the small maths: ratio / necessity / vacuity guards, the split rules (3:1 and explicit)
Part 2  the pass rule, vacuity exclusion and graded companion, checked against merge.py's own pandas rule
Part 3  end to end on a tiny CPU engine (MockModel + MockSAEBank from conftest): the src scorer and specificity
        reproduce the 062 run's experiment code (049 amp_eval_pass_v2.score_circuit(c, {}, skip_roles=True) and
        056 specificity.score) field for field, in both split modes, and the fill budget follows the bank's k
Part 4  the amplifier flag, near-threshold stratum and per-target specificity summary

The same equivalence on the run's own circuits and the H100 rows is a GPU check
(experiments/067-src-scorer-equivalence/check_equivalence.py).
"""
import importlib.util
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from circuit.protocol_contexts import FixedSelector
from circuit.probe_dataset import ProbeDataset
from circuit.types.feature_id import FeatureID
from eval import wcm_eval as W
from eval import wcm_specificity as S
from model.hooks import capture_activations
from store.circuits import Circuit, CircuitNode
from tests.conftest import D_MODEL, MockModel, MockSAEBank

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"


def _load(path, name):
    """An experiment module by path (no sys.path changes); skips when the experiment code is not in the checkout."""
    if not path.exists():
        pytest.skip("experiment code not present: %s" % path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================================== Part 1
class TestMaths:
    def test_ratio_forms_and_guards(self):
        assert W.ratio(3.0, 1.0, 5.0) == 0.5
        assert W.ratio(1.0, 1.0, 5.0) == 0.0
        assert W.ratio(7.0, 1.0, 5.0) == 1.5                    # unclipped
        assert W.ratio(3.0, 5.0, 5.0) is None                   # baseline not below a_pos
        assert W.ratio(3.0, 6.0, 5.0) is None
        assert W.frac(1.0, 2.0, 2.0 + 1e-12) is None            # zero denominator

    def test_necessity(self):
        assert W.necessity(1.0, 5.0, 1.0) == 1.0                # members removed -> back at the empty circuit
        assert W.necessity(5.0, 5.0, 1.0) == 0.0
        assert W.necessity(0.0, 5.0, 1.0) == 1.25
        assert W.necessity(1.0, 5.0, 5.0) is None
        assert W.necessity(1.0, 5.0, 6.0) is None

    def test_vacuous_is_relative_to_a_pos(self):
        assert W.vacuous(1.0, 0.96)                             # |a_pos - e0| = 0.04 < 0.05
        assert not W.vacuous(1.0, 0.94)
        assert not W.vacuous(10.0, 0.0)
        assert not W.vacuous(0.0, 0.0)                          # 0 < 0 is false: the ratio is None instead

    @pytest.mark.parametrize("n, n_tr", [(64, 48), (58, 44), (18, 14), (20, 15), (8, 6), (2, 2), (1, 1)])
    def test_split_n_is_the_engine_rule(self, n, n_tr):
        assert W.split_n(n) == n_tr                             # round half to even: 58 * 0.25 = 14.5 -> 14

    def test_eval_split_modes(self):
        md = dict(n_train=48, n=58)
        assert W.eval_split(64, dict(n_train=48), "explicit") == 48
        assert W.eval_split(58, md, "explicit") == 48           # the padded mid-band probe: held-out stays held out
        assert W.eval_split(58, md, "as_run") == 44             # protocol-v1.md §3.6: 4 train contexts move
        assert W.eval_split(58, None, "as_run") == 44

    @pytest.mark.parametrize("md", [None, {}, dict(n_train=0), dict(n_train=58), dict(n_train=60)])
    def test_explicit_split_needs_both_slices(self, md):
        with pytest.raises(ValueError):
            W.eval_split(58, md, "explicit")

    def test_unknown_split_raises(self):
        with pytest.raises(ValueError):
            W.eval_split(64, dict(n_train=48), "stratified")


# =============================================================================================== Part 2
def _row(z=1.0, a=1.0, c=1.0, nec=0.95, vac=False, read="tk", **kw):
    r = {"free0_" + read: z, "freeM_topk_" + read: a, "freeN_topk_" + read: c, "phi_sup_blind_tk": nec,
         "vacuous_tk": vac}
    r.update(kw)
    return r


class TestPassRule:
    def test_band_and_necessity_edges_inclusive(self):
        assert W.passes(_row(z=0.8, a=1.5, c=1.0, nec=0.9))
        assert not W.passes(_row(z=0.79))
        assert not W.passes(_row(c=1.51))
        assert not W.passes(_row(nec=0.899))

    @pytest.mark.parametrize("missing", [None, float("nan")])
    def test_missing_score_fails(self, missing):
        assert not W.passes(_row(a=missing))
        assert not W.passes(_row(nec=missing))
        assert not W.passes({"phi_sup_blind_tk": 1.0})

    def test_pre_read_keeps_the_tk_necessity(self):
        r = _row(read="pre", nec=0.95, phi_sup_blind_pre=0.1)
        assert W.passes(r, read="pre")                           # pass_rule.py's near-threshold companion
        assert not W.passes(r, read="pre", necessity_read="pre")

    def test_vacuity_flag(self):
        assert W.is_vacuous(_row(vac=True))
        assert not W.is_vacuous(_row(vac=None))
        assert not W.is_vacuous({})

    def test_worst_deviation(self):
        assert W.worst_deviation(_row(z=0.7, a=1.2, c=1.0)) == pytest.approx(0.3)
        assert W.worst_deviation(_row(z=0.9, a=None, c=1.0)) == pytest.approx(0.1)          # merge.py: skipna
        assert W.worst_deviation(_row(z=0.9, a=None, c=1.0), skipna=False) is None           # rescore_mid.py
        assert W.worst_deviation({}) is None

    def test_pass_summary_drops_vacuous(self):
        rows = [_row(), _row(z=0.5), _row(vac=True), _row(a=None)]
        s = W.pass_summary(rows)
        assert s == dict(n=4, n_vacuous=1, n_scored=3, n_pass=1, pass_rate=pytest.approx(1 / 3))

    def test_matches_merge_py(self):
        pd = pytest.importorskip("pandas")
        M = _load(EXP / "062-h100-protocol-v1" / "merge.py", "_merge_062")
        g = torch.Generator().manual_seed(3)
        rows = []
        for j in range(400):
            v = (torch.rand(4, generator=g) * 1.2 + 0.5).tolist()
            r = _row(*v, vac=bool(j % 17 == 0))
            if j % 11 == 0:
                r["freeN_topk_tk"] = None
            rows.append(r)
        df = pd.DataFrame(rows)
        ok = df[~M.vacuous(df)]
        assert list(M.passes(ok)) == [W.passes(r) for r in rows if not W.is_vacuous(r)]
        worst = (ok[M.HEAD] - 1).abs().max(axis=1)
        assert list(worst) == pytest.approx([W.worst_deviation(r) for r in rows if not W.is_vacuous(r)])


# =============================================================================================== Part 3
class EncBank(MockSAEBank):
    """MockSAEBank with a tied, unit-norm dictionary (so a code reconstructs part of the stream and ablations move
    the target by graded amounts) plus the encoder surface the target read uses (encoder.weight, _get_bias_eff)."""

    def __init__(self, k=4, d_sae=32):
        super().__init__(d_sae=d_sae, k=k)
        g = torch.Generator().manual_seed(11)
        for kind in self.kinds:
            for m in self.saes[kind]:
                m.W_enc = m.W_enc / m.W_enc.norm(dim=1, keepdim=True)
                m.W_dec = m.W_enc.T.clone()
                m.b_enc = -0.2 * torch.rand(m.d_sae, generator=g)
                m.decoder_bias = m.decoder_bias * 0.1
                m.encoder = SimpleNamespace(weight=m.W_enc)
                m._get_bias_eff = (lambda m=m: m.b_enc)


class TinyInference:
    """Inference.forward's contract on MockModel: token embedding, then the activation callback hooks registered
    before the patcher's (as the engine registers them)."""

    def __init__(self, n_vocab=64, seed=5):
        g = torch.Generator().manual_seed(seed)
        self.model = MockModel()
        self.emb = torch.randn(n_vocab, D_MODEL, generator=g)
        self.n_forward = 0

    def disable_compile(self):
        pass

    def enable_compile(self):
        pass

    def forward(self, tokens, activations_callback=None, patcher=None, return_activations=True, **kw):
        self.n_forward += 1
        x = self.emb[tokens.long()]
        with capture_activations(self.model, callback=activations_callback, capture=False), \
                (patcher(self.model) if patcher else nullcontext()):
            self.model(x)
        return tokens, None, None


SEED_SITE = (1, "resid")


def _engine(k=4):
    return TinyInference(), EncBank(k=k)


def _target_and_contexts(inf, bank, n_pos=20, n_train=15, n_neg=16, n_seq_pool=240, T=8):
    """Target = the (1, resid) latent with the largest mean peak; activating contexts = every 6th of its strongest
    sequences (MockModel has no cross-position mixing, so the very strongest share one peak token), anchor = argmax;
    contrast = its weakest. ProbeDataset metadata carries n_train (protocol_contexts form)."""
    from sae.dense import sparse_topk_to_dense
    g = torch.Generator().manual_seed(7)
    toks = torch.randint(0, inf.emb.shape[0], (n_seq_pool, T), generator=g)
    codes = []
    k2i = {k: i for i, k in enumerate(bank.kinds)}

    def hook(layer_idx, acts):
        if layer_idx == SEED_SITE[0]:
            ta, ti = bank.encode(acts[k2i[SEED_SITE[1]]], SEED_SITE[1], layer_idx)
            codes.append(sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=torch.float32))

    inf.forward(toks, activations_callback=hook)
    dense = codes[0]                                            # [N, T, D]
    peak = dense.max(dim=1).values                              # [N, D]
    latent = int(peak.mean(0).argmax())
    order = torch.argsort(peak[:, latent], descending=True)
    pos = order[:6 * n_pos:6]
    neg = order.flip(0)[:n_neg]
    probe = ProbeDataset(pos_tokens=toks[pos], target_tokens=toks[pos], neg_tokens=toks[neg],
                         pos_argmax=dense[pos, :, latent].argmax(dim=1), metadata=dict(n_train=n_train, n=n_pos))
    return latent, probe


def _circuit(inf, bank, latent, probe, per_site=3, amplitude=True):
    """Members: the per_site latents with the largest mean activation on the activating contexts at every upstream
    site, in a scrambled node order, with fitted-looking amplitudes."""
    from sae.dense import sparse_topk_to_dense
    ups = sorted(W.upstream_sites(bank, *SEED_SITE))
    means = {}
    k2i = {k: i for i, k in enumerate(bank.kinds)}

    def hook(layer_idx, acts):
        for kd in bank.kinds:
            if (layer_idx, kd) in ups:
                ta, ti = bank.encode(acts[k2i[kd]], kd, layer_idx)
                means[(layer_idx, kd)] = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=torch.float32).mean((0, 1))

    inf.forward(probe.pos_tokens, activations_callback=hook)
    g = torch.Generator().manual_seed(13)
    members = [(s, int(i)) for s in ups for i in torch.argsort(means[s], descending=True)[:per_site].tolist()]
    members = [members[j] for j in torch.randperm(len(members), generator=g).tolist()]
    c = Circuit(name="t", metadata=dict(amp_stats=dict(median=0.9)))
    c.add_node(CircuitNode(metadata={"feature_id": FeatureID(SEED_SITE[0], SEED_SITE[1], latent), "role": "seed"}))
    for (l, kd), i in members:
        md = {"feature_id": FeatureID(l, kd, i), "role": "ablation_support"}
        if amplitude:
            md["amplitude"] = float(0.2 + 1.6 * torch.rand(1, generator=g))
        c.add_node(CircuitNode(metadata=md))
    return c


def _experiment_scorer(monkeypatch, inf, bank, probe, split):
    V = _load(EXP / "049-circuit-graph" / "amp_eval_pass_v2.py", "_amp_eval_pass_v2")
    M0 = SimpleNamespace(build_probe_dataset=lambda comp, i: probe,
                         _neg_context_selector=lambda: FixedSelector(probe.neg_tokens))
    monkeypatch.setattr(V, "G", dict(inference=inf, bank=bank, M0=M0, avg_acts=None, KINDS=list(bank.kinds),
                                     NK=len(bank.kinds), D=bank.d_sae, K=int(bank.k), device=torch.device("cpu")))
    monkeypatch.setattr(V, "CTR_SOURCE", "close")              # the 062 driver's setting
    monkeypatch.setattr(V, "EVAL_SPLIT", split)
    monkeypatch.setattr(V, "neg_ctx_store_fallback", lambda comp, sl: False)
    return V


def _assert_rows_equal(a, b):
    assert set(a) == set(b), "fields differ: %s" % sorted(set(a) ^ set(b))
    for f in a:
        if f in W.TIMING_FIELDS:
            continue
        assert a[f] == b[f] or (isinstance(a[f], float) and math.isnan(a[f]) and math.isnan(b[f])), f


@pytest.fixture
def setup_scored():
    inf, bank = _engine()
    latent, probe = _target_and_contexts(inf, bank)
    return inf, bank, latent, probe, _circuit(inf, bank, latent, probe)


class TestScorerEquivalence:
    @pytest.mark.parametrize("split", ["as_run", "explicit"])
    def test_matches_experiment_scorer(self, monkeypatch, setup_scored, split):
        inf, bank, latent, probe, c = setup_scored
        V = _experiment_scorer(monkeypatch, inf, bank, probe, split)
        exp = V.score_circuit(c, {}, skip_roles=True)
        src = W.score_circuit(inf, bank, c, probe, split=split, ctr_fallback=False)
        _assert_rows_equal(exp, src)

    @pytest.mark.parametrize("split", ["as_run", "explicit"])
    def test_padded_probe_matches_and_split_differs(self, monkeypatch, split):
        """A short probe (18 rows, n_train 15): the run's 3:1 rule takes 14 train rows (§3.6), explicit takes 15."""
        inf, bank = _engine()
        latent, probe = _target_and_contexts(inf, bank, n_pos=18, n_train=15)
        c = _circuit(inf, bank, latent, probe)
        V = _experiment_scorer(monkeypatch, inf, bank, probe, split)
        exp = V.score_circuit(c, {}, skip_roles=True)
        src = W.score_circuit(inf, bank, c, probe, split=split, ctr_fallback=False)
        _assert_rows_equal(exp, src)
        assert (src["n_train"], src["n_ho"]) == ((14, 4) if split == "as_run" else (15, 3))

    def test_unweighted_and_override_alphas(self, monkeypatch):
        inf, bank = _engine()
        latent, probe = _target_and_contexts(inf, bank)
        c = _circuit(inf, bank, latent, probe, amplitude=False)
        V = _experiment_scorer(monkeypatch, inf, bank, probe, "as_run")
        _assert_rows_equal(V.score_circuit(c, {}, skip_roles=True),
                           W.score_circuit(inf, bank, c, probe, split="as_run", ctr_fallback=False))
        ones = {s: {i: 1.0 for i in d} for s, d in W.circuit_alphas(_circuit(inf, bank, latent, probe)).items()}
        _assert_rows_equal(V.score_circuit(c, {}, override_alphas=ones, skip_roles=True),
                           W.score_circuit(inf, bank, c, probe, split="as_run", alphas=ones, ctr_fallback=False))

    def test_matches_experiment_specificity(self, monkeypatch, setup_scored):
        inf, bank, latent, probe, c = setup_scored
        V = _experiment_scorer(monkeypatch, inf, bank, probe, "as_run")
        Sx = _load(EXP / "056-specificity" / "specificity.py", "_specificity_056")
        CleanTap, CircuitTap = Sx.make_patchers(V.G)
        exp = Sx.score(V.G, c, V, CleanTap, CircuitTap, "B")
        src = S.specificity_rows(inf, bank, c, probe, arm="B", split="as_run")
        assert len(exp) == len(src) == 3
        for a, b in zip(exp, src):
            _assert_rows_equal(a, b)


class TestScorerRow:
    def test_protocol_invariants(self, setup_scored):
        inf, bank, latent, probe, c = setup_scored
        r = W.score_circuit(inf, bank, c, probe)
        assert r["all_sites_ablated"] is True
        assert (r["n_train"], r["n_ho"], r["n_neg_train"], r["n_neg_ho"]) == (15, 5, 12, 4)
        assert r["a_pos_tk"] > r["e0_tk"] and r["free0_rand_tk"] == 0.0          # free0_rand = the empty circuit
        assert r["free0_rand_raw_tk"] == r["e0_tk"]
        for rd in W.READS:                                                       # roles not computed (DAN-72)
            assert r["phi_sup_role_" + rd] == r["phi_sup_blind_" + rd] == r["sup_activators_only_" + rd]
            assert r["release_" + rd] == 0.0
        assert (r["n_activator"], r["n_inhibitor"], r["inhibitor_mass_share"]) == (r["n"], 0, None)
        assert "phi_pin_alpha_zero_tk" in r and r["n"] == 15 and r["n_member_sites"] == 5
        assert r["thin"] is True and r["ctr_source"] == "close" and r["ctr_fallback"] is None

    def test_all_members_at_alpha_one_is_the_natural_run(self):
        """Keeping every latent at every upstream site with alpha = 1 is the identity edit: free0 = 1 exactly."""
        inf, bank = _engine()
        latent, probe = _target_and_contexts(inf, bank)
        c = _circuit(inf, bank, latent, probe, per_site=bank.d_sae, amplitude=False)
        r = W.score_circuit(inf, bank, c, probe)
        assert r["free0_raw_tk"] == pytest.approx(r["a_pos_tk"], rel=1e-5)
        assert r["free0_tk"] == pytest.approx(1.0, rel=1e-5)

    def test_fill_budget_is_the_banks_k(self, setup_scored):
        inf, bank, latent, probe, c = setup_scored
        r = W.score_circuit(inf, bank, c, probe)
        assert W.score_circuit(inf, bank, c, probe, topk=bank.k)["eM_topk_pre"] == r["eM_topk_pre"]
        wide = W.score_circuit(inf, bank, c, probe, topk=4 * bank.k)
        assert wide["eM_topk_pre"] != r["eM_topk_pre"]
        assert wide["e0_pre"] == r["e0_pre"]                                     # zero / dense fills ignore k

    def test_skip_rows(self, setup_scored):
        inf, bank, latent, probe, c = setup_scored
        thin = ProbeDataset(probe.pos_tokens[:7], probe.target_tokens[:7], probe.neg_tokens, probe.pos_argmax[:7],
                            dict(n_train=5))
        assert W.score_circuit(inf, bank, c, thin) == dict(seed="1.resid.%d" % latent, skip="thin_probes", n=15,
                                                           n_pos=7)
        assert W.score_circuit(inf, bank, c, None)["skip"] == "thin_probes"
        r = W.score_circuit(inf, bank, c, probe, probe.neg_tokens[:1])
        assert (r["skip"], r["n_neg"]) == ("no_negatives", 1)

    def test_contrast_tokens_override_probe_negatives(self, setup_scored):
        inf, bank, latent, probe, c = setup_scored
        a = W.score_circuit(inf, bank, c, probe)
        b = W.score_circuit(inf, bank, c, probe, probe.neg_tokens.flip(0))
        assert a["a_pos_tk"] == b["a_pos_tk"] and a["a_base_pre"] != b["a_base_pre"]


# =============================================================================================== Part 4
class TestSpecificityFlags:
    def test_amplifier_flag(self):
        assert S.amplifier_flag(dict(target_faith_pre=0.5, sibling_faith_median=0.4))
        assert S.amplifier_flag(dict(target_faith_pre=0.9, sibling_faith_median=0.73))
        assert not S.amplifier_flag(dict(target_faith_pre=0.9, sibling_faith_median=0.71))
        assert not S.amplifier_flag(dict(target_faith_pre=0.49, sibling_faith_median=1.0))
        assert not S.amplifier_flag(dict(target_faith_pre=0.9, sibling_faith_median=None))
        assert not S.amplifier_flag(dict(target_faith_pre=float("nan"), sibling_faith_median=1.0))

    @pytest.mark.parametrize("k, rank, near", [(128, 64, True), (128, 63, False), (32, 16, True), (32, 15, False),
                                               (16, 8, True), (16, 7.5, False), (16, None, False)])
    def test_near_threshold_is_per_model(self, k, rank, near):
        assert S.near_threshold(rank, k) is near

    def test_target_specificity(self):
        rows = [dict(pi=p, K=32, rank_clean=20.0, target_faith_pre=t, sibling_faith_median=s, switched_on_circuit=w)
                for p, t, s, w in (("Z", 0.9, 0.2, 1.0), ("A", 0.9, 0.8, 3.0), ("C", 0.4, None, 2.0))]
        t = S.target_specificity(rows)
        assert t == dict(amp_any=True, sib_C=pytest.approx(0.5), rank_clean=20.0, lifted_C=2.0, near_threshold=True)
        assert S.target_specificity(rows, k=128)["near_threshold"] is False
        with pytest.raises(ValueError):
            S.target_specificity(rows + [dict(rows[0], K=128)])

    def test_latent_faith_guard(self):
        clean = torch.tensor([[2.0, 1.0, 0.01]])
        empty = torch.tensor([[0.0, 0.98, 0.0]])
        circ = torch.tensor([[1.0, 1.0, 0.0]])
        f = S.latent_faith(circ, empty, clean)
        assert f[0] == 0.5 and math.isnan(f[1]) and f[2] == 0.0
