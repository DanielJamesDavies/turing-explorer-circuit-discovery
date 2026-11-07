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
        assert W.eval_split(58, md, "as_run") == 44             # protocol.md §3.6: 4 train contexts move
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
        assert W.passes(_row(z=0.75, a=1.5, c=1.0, nec=0.9))
        assert not W.passes(_row(z=0.749))
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
        assert W.is_vacuous(_row(vac=True))                      # no baselines stored: the v1 flag
        assert not W.is_vacuous(_row(vac=None))
        assert not W.is_vacuous({})
        assert W.is_vacuous(_row(vac=True), rule="v1") and not W.is_vacuous(_row(), rule="v1")
        with pytest.raises(ValueError):
            W.is_vacuous(_row(), rule="v3")

    @pytest.mark.parametrize("ap,e0,em,en,want", [
        (1.0, 0.0, 0.5, 0.5, False),
        (1.0, 0.96, 0.0, 0.0, True),         # v1's case: e0 within 5% below a_pos
        (1.0, 1.2, 0.0, 0.0, True),          # e0 far above a_pos (v1's two-sided flag misses it)
        (1.0, 0.0, 1.0, 0.0, True),          # A baseline reaches a_pos: A undefined
        (1.0, 0.0, 0.0, 0.96, True),         # C denominator positive but below 5% of a_pos
        (1.0, 0.0, 0.0, 0.94, False),
        (1.0, -0.5, -0.5, -0.5, False),      # a negative baseline is a wide denominator
        (0.0, -1.0, -1.0, -1.0, True),       # a_pos <= 0
        (-1.0, -2.0, -2.0, -2.0, True),
    ])
    def test_vacuous_v2(self, ap, e0, em, en, want):
        assert W.vacuous_v2(ap, [e0, em, en]) is want
        r = _row(vac=False, a_pos_tk=ap, e0_tk=e0, eM_topk_tk=em, eN_topk_tk=en)
        assert W.is_vacuous(r) is want and W.is_vacuous(r, read="pre") is want   # one set for both reads
        assert W.is_vacuous(dict(r, vacuous_v2=not want)) is (not want)         # the stored flag wins

    def test_vacuous_v2_falls_back_to_v1_when_a_read_is_missing(self):
        r = _row(vac=True, a_pos_tk=1.0, e0_tk=0.0, eM_topk_tk=None, eN_topk_tk=0.0)
        assert W.is_vacuous(r) and not W.is_vacuous(dict(r, vacuous_tk=False))
        assert not W.is_vacuous(dict(r, vacuous_tk=False, eM_topk_tk=float("nan")))

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

    def test_vacuous_v2_matches_merge_and_083(self):
        """merge.py (pandas) and 083 cmp_common re-implement §12.4: same set as is_vacuous on random rows, with
        missing reads (v1 fallback) and stored flags mixed in."""
        pd = pytest.importorskip("pandas")
        M = _load(EXP / "062-h100-protocol-v1" / "merge.py", "_merge_062")
        C = _load(EXP / "083-v1-comparison" / "cmp_common.py", "_cmp_083")
        g = torch.Generator().manual_seed(5)
        rows = []
        for j in range(600):
            ap, e0, em, en = (torch.rand(4, generator=g) * 1.4 - 0.2).tolist()
            r = _row(vac=bool(j % 13 == 0), a_pos_tk=ap, e0_tk=e0, eM_topk_tk=em, eN_topk_tk=en)
            if j % 7 == 0:
                r["eM_topk_tk"] = None
            if j % 19 == 0:
                r["vacuous_v2"] = bool(j % 2)
            rows.append(r)
        want = [W.is_vacuous(r) for r in rows]
        assert 0 < sum(want) < len(rows)
        assert list(M.vacuous(pd.DataFrame(rows))) == want
        assert [C.vacuous(r) for r in rows] == want
        assert list(M.vacuous(pd.DataFrame(rows), rule="v1")) == [W.is_vacuous(r, rule="v1") for r in rows]


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


SRC_ONLY_FIELDS = {"vacuous_v2"}        # added in src after the 049 scorer was frozen (protocol §12.4)


def _assert_rows_equal(a, b):
    a, b = ({k: v for k, v in r.items() if k not in SRC_ONLY_FIELDS} for r in (a, b))
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
        assert r["vacuous_v2"] == W.vacuous_v2(r["a_pos_tk"], [r[k + "_tk"] for k in W.VACUOUS_BASELINES])
        assert W.is_vacuous(r) == r["vacuous_v2"] == W.is_vacuous({k: v for k, v in r.items() if k != "vacuous_v2"})

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


def _with_contrast(probe, neg, n_neg_train):
    md = dict(probe.metadata)
    if n_neg_train is not None:
        md["n_neg_train"] = n_neg_train
    return ProbeDataset(probe.pos_tokens, probe.target_tokens, neg, probe.pos_argmax, md)


class TestRebuiltContrast:
    """086: a record with a rebuilt (deduplicated, possibly shorter) contrast held-out set reaches the scorers as the
    run's contrast-train rows + the new held-out rows with metadata n_neg_train (protocol_contexts.eval_contrast).
    Without it every scorer is unchanged; with it the contrast-train slice is the run's, whatever the held-out size."""

    def test_default_path_unchanged(self, setup_scored):
        inf, bank, latent, probe, c = setup_scored
        assert "n_neg_train" not in probe.metadata
        a = W.score_circuit(inf, bank, c, probe)
        b = W.score_circuit(inf, bank, c, _with_contrast(probe, probe.neg_tokens, W.split_n(16)))
        _assert_rows_equal(a, b)                         # a full 16-row held-out set: the same split, the same row

    def test_short_contrast_held_out_set(self, monkeypatch, setup_scored):
        inf, bank, latent, probe, c = setup_scored
        nt = probe.neg_tokens                              # 16 rows: 12 train, 4 held out
        short = _with_contrast(probe, torch.cat([nt[:12], nt[12:14]]), 12)
        a, b = W.score_circuit(inf, bank, c, probe), W.score_circuit(inf, bank, c, short)
        assert (b["n_neg_train"], b["n_neg_ho"]) == (12, 2) and (a["n_neg_train"], a["n_neg_ho"]) == (12, 4)
        for f in ("eN_dense_pre", "eN_topk_pre", "eN_topk_tk", "a_pos_tk", "free0_tk", "freeM_topk_tk"):
            assert a[f] == b[f], f                         # C means come from the same 12 contrast-train rows
        naive = W.score_circuit(inf, bank, c, _with_contrast(probe, short.neg_tokens, None))
        assert naive["n_neg_train"] == W.split_n(14) == 10  # what the metadata prevents
        V = _experiment_scorer(monkeypatch, inf, bank, short, "explicit")
        _assert_rows_equal(V.score_circuit(c, {}, skip_roles=True),
                           W.score_circuit(inf, bank, c, short, split="explicit", ctr_fallback=False))
        Sx = _load(EXP / "056-specificity" / "specificity.py", "_specificity_056")
        CleanTap, CircuitTap = Sx.make_patchers(V.G)
        exp = Sx.score(V.G, c, V, CleanTap, CircuitTap, "B")
        src = S.specificity_rows(inf, bank, c, short, arm="B", split="explicit")
        full = S.specificity_rows(inf, bank, c, probe, arm="B", split="explicit")
        for x, y, z in zip(exp, src, full):
            _assert_rows_equal(x, y)
            _assert_rows_equal(y, z)                       # specificity reads only the contrast-train rows

    def test_explicit_contrast_tokens_ignore_the_metadata(self, setup_scored):
        inf, bank, latent, probe, c = setup_scored
        short = _with_contrast(probe, probe.neg_tokens[:14], 12)
        r = W.score_circuit(inf, bank, c, short, probe.neg_tokens)   # caller's own contrast rows: split_n, as before
        assert (r["n_neg_train"], r["n_neg_ho"]) == (12, 4)
        with pytest.raises(ValueError):
            W.score_circuit(inf, bank, c, _with_contrast(probe, probe.neg_tokens[:12], 12))


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


# =============================================================================================== Part 5
# The output-endpoint scorer (score_output, experiment 080): the same circuit-only runs read at the model's output.
class TinyLogitInference(TinyInference):
    """TinyInference with an unembedding: forward returns logits = final stream @ W_U (Inference.forward's
    (tokens, logits, acts) contract)."""

    def __init__(self, n_vocab=64, seed=5):
        super().__init__(n_vocab, seed)
        g = torch.Generator().manual_seed(seed + 100)
        self.W_U = torch.randn(D_MODEL, n_vocab, generator=g)

    def forward(self, tokens, activations_callback=None, patcher=None, return_activations=True, **kw):
        self.n_forward += 1
        x = self.emb[tokens.long()]
        with capture_activations(self.model, callback=activations_callback, capture=False), \
                (patcher(self.model) if patcher else nullcontext()):
            y = self.model(x)
        return tokens, y @ self.W_U, None


def _output_setup(per_site=3, amplitude=True):
    """Target, contexts and a circuit as in Part 3. The probe's "next tokens" are, at every position, the token whose
    log-prob the zero ablation of the scope lowers most (argmax of log p_full - log p_empty, which is >= 0 because
    both distributions sum to 1), so every output denominator is positive. In this random toy the full model's own
    top-1 is often RAISED by the ablation."""
    inf, bank = TinyLogitInference(), EncBank()
    latent, probe = _target_and_contexts(inf, bank)
    pt = probe.pos_tokens
    full = torch.log_softmax(inf.forward(pt)[1], -1)
    empty = W.EditTrackingCircuitOnly(bank=bank, keep_indices={}, in_scope=set(W.upstream_sites(bank, *SEED_SITE)),
                                      seed_layer=-1, seed_kind="", seed_latent_idx=0)
    drop = full - torch.log_softmax(inf.forward(pt, patcher=empty)[1], -1)
    probe = ProbeDataset(pt, drop.argmax(-1), probe.neg_tokens, probe.pos_argmax, dict(probe.metadata))
    return inf, bank, latent, probe, _circuit(inf, bank, latent, probe, per_site=per_site, amplitude=amplitude)


def _seed_only(latent):
    c = Circuit(name="empty")
    c.add_node(CircuitNode(metadata={"feature_id": FeatureID(SEED_SITE[0], SEED_SITE[1], latent), "role": "seed"}))
    return c


class TestOutputScorer:
    def test_every_upstream_latent_at_alpha_one_is_the_full_model(self):
        """Keeping every latent at every upstream site with alpha = 1 is the identity edit: output Z = 1; removing
        all of them from the live model is the zero empty circuit: output necessity = 1."""
        inf, bank, latent, probe, _ = _output_setup()
        c = _circuit(inf, bank, latent, probe, per_site=bank.d_sae, amplitude=False)
        r = W.score_output(inf, bank, c, probe)
        assert r["out_e0"] < r["out_full"] and not r["out_vacuous"]
        assert r["out_free0_raw"] == pytest.approx(r["out_full"], rel=1e-5)
        assert r["out_free0"] == pytest.approx(1.0, rel=1e-5)
        assert r["out_free0_a1"] == pytest.approx(1.0, rel=1e-5)
        assert r["out_phi_sup_blind"] == pytest.approx(1.0, rel=1e-5)
        assert r["n"] == 5 * bank.d_sae and r["out_all_sites_ablated"] is True

    def test_empty_circuit_scores_exactly_zero(self):
        inf, bank, latent, probe, _ = _output_setup()
        r = W.score_output(inf, bank, _seed_only(latent), probe)
        for f in W.OUT_FAITH + ("out_free0_a1", W.OUT_NECESSITY):
            assert r[f] == 0.0, f
        assert r["n"] == 0 and r["out_all_sites_ablated"] is True

    def test_scope_is_upstream_and_the_target_site_runs_live(self, monkeypatch):
        inf, bank, latent, probe, c = _output_setup()
        seen = []
        orig = W.EditTrackingCircuitOnly.transform

        def spy(self, layer_idx, kind, x):
            out = orig(self, layer_idx, kind, x)
            seen.append(((layer_idx, kind), (layer_idx, kind) in self.in_scope))
            return out
        monkeypatch.setattr(W.EditTrackingCircuitOnly, "transform", spy)
        r = W.score_output(inf, bank, c, probe)
        assert r["out_n_scope_sites"] == 5 and r["out_all_sites_ablated"] is True
        assert {s for s, ins in seen if ins} == set(W.upstream_sites(bank, *SEED_SITE))
        assert (SEED_SITE, False) in seen                       # reached, not ablated: recomputed live

    def test_whole_model_scope_also_ablates_the_target_site(self):
        inf, bank, latent, probe, c = _output_setup()
        up = W.score_output(inf, bank, c, probe)
        al = W.score_output(inf, bank, c, probe, scope="all")
        assert al["out_n_scope_sites"] == 6 and al["out_scope"] == "all"
        assert al["out_full"] == up["out_full"] and al["out_e0"] != up["out_e0"]
        with pytest.raises(ValueError, match="scope"):
            W.score_output(inf, bank, c, probe, scope="downstream")

    def test_members_outside_the_scope_are_dropped_and_counted(self):
        inf, bank, latent, probe, c = _output_setup()
        base = W.score_output(inf, bank, c, probe)
        c.add_node(CircuitNode(metadata={"feature_id": FeatureID(SEED_SITE[0], SEED_SITE[1], (latent + 1) % 32),
                                         "role": "ablation_support", "amplitude": 1.0}))
        r = W.score_output(inf, bank, c, probe)
        assert r["out_n_out_of_scope"] == 1 and r["n"] == base["n"]
        assert r["out_free0"] == base["out_free0"] and r["out_phi_sup_blind"] == base["out_phi_sup_blind"]

    def test_split_and_counts_mirror_the_latent_scorer(self):
        inf, bank, latent, probe, c = _output_setup()
        lat, out = W.score_circuit(inf, bank, c, probe), W.score_output(inf, bank, c, probe)
        assert (out["out_n_train"], out["out_n_ho"], out["out_n_neg_train"]) == (lat["n_train"], lat["n_ho"],
                                                                                lat["n_neg_train"])
        assert out["n"] == lat["n"] and out["seed"] == lat["seed"]
        r = W.score_output(inf, bank, c, probe, split="as_run")
        assert r["out_n_train"] == W.split_n(probe.pos_tokens.shape[0])

    def test_coefficients_and_override(self):
        """Unweighted circuit: Z at the fitted gains == Z at alpha = 1. Weighted circuit scored with alphas = 1 ==
        its own out_free0_a1."""
        inf, bank, latent, probe, c = _output_setup(amplitude=False)
        r = W.score_output(inf, bank, c, probe)
        assert r["out_free0"] == r["out_free0_a1"]
        inf, bank, latent, probe, c = _output_setup()
        fitted = W.score_output(inf, bank, c, probe)
        ones = {s: {i: 1.0 for i in d} for s, d in W.circuit_alphas(c).items()}
        assert W.score_output(inf, bank, c, probe, alphas=ones)["out_free0"] == fitted["out_free0_a1"]
        assert fitted["out_free0"] != fitted["out_free0_a1"]

    def test_skip_rows(self):
        inf, bank, latent, probe, c = _output_setup()
        thin = ProbeDataset(probe.pos_tokens[:7], probe.target_tokens[:7], probe.neg_tokens, probe.pos_argmax[:7],
                            dict(n_train=5))
        assert W.score_output(inf, bank, c, thin)["skip"] == "thin_probes"
        assert W.score_output(inf, bank, c, probe, probe.neg_tokens[:1])["skip"] == "no_negatives"

    def test_next_tokens_are_gathered_at_the_anchor(self):
        tt = torch.arange(12).reshape(3, 4)
        assert W.next_tokens_at(SimpleNamespace(target_tokens=tt), torch.tensor([0, 3, 9]), 3).tolist() == [0, 7, 11]
        flat = torch.tensor([5, 6, 7])
        assert W.next_tokens_at(SimpleNamespace(target_tokens=flat), torch.tensor([1, 1, 1]), 3).tolist() == [5, 6, 7]

    def test_vacuity_ratio_guard_and_pass_rule(self):
        assert W.output_vacuous(-1.0, -1.05) and W.output_vacuous(-1.0, -0.95)
        assert not W.output_vacuous(-1.0, -1.2)
        assert W.ratio(-2.0, -3.0, -1.0) == 0.5 and W.ratio(-2.0, -1.0, -1.0) is None      # the shared guard
        row = {"out_free0": 0.8, "out_freeM_topk": 1.5, "out_freeN_topk": 1.0, "out_phi_sup_blind": 0.9}
        assert W.passes_output(row)
        assert not W.passes_output(dict(row, out_freeN_topk=0.749))
        assert not W.passes_output(dict(row, out_phi_sup_blind=None))
        assert not W.passes_output(dict(row, out_freeM_topk=float("nan")))


# =============================================================================================== Part 6
# The task-endpoint scorer (score_task, experiment 081): task metrics at each prompt's anchor, every site in scope,
# right-padded prompts with valid-position masks.
def _task_setup(per_site=3, amplitude=True):
    """Prompts = Part 3's activating contexts (15 train + 5 held out), counterfactuals = its contrast contexts, anchor =
    the target's argmax. good / bad = per row the token whose log-prob the zero ablation of every site lowers most /
    raises most at the anchor, so the logit-diff denominator is positive on every row."""
    inf, bank = TinyLogitInference(), EncBank()
    latent, probe = _target_and_contexts(inf, bank)
    pt, pa = probe.pos_tokens, probe.pos_argmax
    allsites = W.scope_sites(bank, 0, bank.kinds[0], "all")
    full = torch.log_softmax(inf.forward(pt)[1], -1)
    empty = W.EditTrackingCircuitOnly(bank=bank, keep_indices={}, in_scope=allsites, seed_layer=-1, seed_kind="",
                                      seed_latent_idx=0)
    drop = (full - torch.log_softmax(inf.forward(pt, patcher=empty)[1], -1))[torch.arange(pt.shape[0]), pa]
    good, bad = drop.argmax(-1), drop.argmin(-1)
    alphas = W.circuit_alphas(_circuit(inf, bank, latent, probe, per_site=per_site, amplitude=amplitude))
    return inf, bank, alphas, pt, pa, probe.neg_tokens, good, bad


class TestTaskScorer:
    def test_metric_kinds_on_hand_logits(self):
        lp = torch.log_softmax(torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, 0.0, 3.0, 0.0]]), -1)
        good, bad = torch.tensor([0, 1]), torch.tensor([1, 2])
        assert W.task_metric(lp, "logit_diff", good, bad).tolist() == pytest.approx([1.0, -3.0])
        assert W.task_metric(lp, "logprob", good).tolist() == pytest.approx([float(lp[0, 0]), float(lp[1, 1])])
        sg, sb = torch.tensor([[0, 1], [2, -1]]), torch.tensor([[2, 3], [0, 1]])
        p = lp.exp()
        assert W.task_metric(lp, "set_probdiff", sg, sb).tolist() == pytest.approx(
            [float(p[0, :2].sum() - p[0, 2:].sum()), float(p[1, 2] - p[1, :2].sum())], rel=1e-5)
        assert W.task_metric(lp, "set_logmass", sg, sb).tolist() == pytest.approx(
            [math.log(float(p[0, :2].sum() / p[0, 2:].sum())), math.log(float(p[1, 2] / p[1, :2].sum()))], rel=1e-5)
        # a one-token set is the token metric
        assert W.task_metric(lp, "set_logmass", good[:, None], bad[:, None]).tolist() == pytest.approx(
            W.task_metric(lp, "logit_diff", good, bad).tolist())
        assert W.task_correct(lp, "logit_diff", good, bad).tolist() == [True, False]
        assert W.task_correct(lp, "logprob", torch.tensor([0, 1])).tolist() == [True, False]
        assert W.task_top1(lp, sg).tolist() == [True, True]
        with pytest.raises(ValueError, match="kind"):
            W.task_metric(lp, "accuracy", good, bad)
        with pytest.raises(ValueError, match="bad"):
            W.task_metric(lp, "logit_diff", good)

    def test_every_latent_at_alpha_one_is_the_full_model_and_empty_is_zero(self):
        inf, bank, _, pt, pa, cf, good, bad = _task_setup()
        allsites = W.scope_sites(bank, 0, bank.kinds[0], "all")
        everything = {s: {i: 1.0 for i in range(bank.d_sae)} for s in allsites}
        m = {"task": ("logit_diff", good, bad)}
        r = W.score_task(inf, bank, everything, pt, pa, 15, cf[:12], m)
        assert r["task_e0"] < r["task_full"] and not r["task_vacuous"]
        assert r["task_free0"] == pytest.approx(1.0, rel=1e-5) and r["task_nec"] == pytest.approx(1.0, rel=1e-5)
        assert r["task_acc_free0"] == r["task_acc_full"] and r["all_sites_ablated"] is True
        assert r["n"] == len(allsites) * bank.d_sae and r["n_sites_scope"] == 6 and (r["n_train"], r["n_ho"]) == (15, 5)
        e = W.score_task(inf, bank, {}, pt, pa, 15, cf[:12], m)
        for f in ("task_free0", "task_freeM_topk", "task_freeN_topk", "task_free0_a1", "task_nec"):
            assert e[f] == 0.0, f
        assert e["n"] == 0 and not e["task_pass"]

    def test_coefficients_are_applied_and_alpha_one_is_recorded(self):
        inf, bank, alphas, pt, pa, cf, good, bad = _task_setup()
        m = {"task": ("logit_diff", good, bad)}
        fitted = W.score_task(inf, bank, alphas, pt, pa, 15, cf[:12], m)
        ones = {s: {i: 1.0 for i in d} for s, d in alphas.items()}
        a1 = W.score_task(inf, bank, ones, pt, pa, 15, cf[:12], m)
        assert a1["task_free0"] == pytest.approx(fitted["task_free0_a1"], rel=1e-6)
        assert fitted["task_free0"] != pytest.approx(fitted["task_free0_a1"])
        assert a1["task_free0"] == pytest.approx(a1["task_free0_a1"], rel=1e-6)
        # necessity does not depend on the coefficients (members are SET to 0 in the live model)
        assert fitted["task_nec"] == pytest.approx(a1["task_nec"], rel=1e-6)

    def test_several_metrics_share_the_runs(self):
        inf, bank, alphas, pt, pa, cf, good, bad = _task_setup()
        r = W.score_task(inf, bank, alphas, pt, pa, 15, cf[:12],
                         {"task": ("logit_diff", good, bad), "obj": ("set_logmass", good[:, None], bad[:, None])},
                         vacuous_abs={"task": 1e9})
        for f in ("free0", "freeM_topk", "freeN_topk", "nec", "full", "e0"):
            assert r["task_" + f] == pytest.approx(r["obj_" + f], rel=1e-5), f
        assert r["task_vacuous"] and not r["obj_vacuous"]           # per-metric vacuity thresholds

    def test_valid_masks_make_right_padding_inert(self):
        """Appending padding columns after every prompt changes nothing once the A / C means skip them (the toy model
        has no cross-position mixing, so the anchor read itself is unaffected either way); without the masks the
        padding columns enter the means."""
        inf, bank, alphas, pt, pa, cf, good, bad = _task_setup()
        m = {"task": ("logit_diff", good, bad)}
        base = W.score_task(inf, bank, alphas, pt, pa, 15, cf[:12], m)
        pad = torch.zeros(pt.shape[0], 5, dtype=pt.dtype)
        ptp, cfp = torch.cat([pt, pad], 1), torch.cat([cf[:12], pad[:12]], 1)
        v = torch.zeros(ptp.shape, dtype=torch.bool)
        v[:, :pt.shape[1]] = True
        masked = W.score_task(inf, bank, alphas, ptp, pa, 15, cfp, m, valid=v, cf_valid=v[:12])
        for f in ("task_freeM_topk", "task_freeN_topk", "task_eM_topk", "task_eN_topk", "task_free0"):
            assert masked[f] == pytest.approx(base[f], rel=1e-5), f
        unmasked = W.score_task(inf, bank, alphas, ptp, pa, 15, cfp, m)
        assert unmasked["task_eM_topk"] != pytest.approx(base["task_eM_topk"], rel=1e-6)
        assert unmasked["task_free0"] == pytest.approx(base["task_free0"], rel=1e-5)      # Z has no means

    def test_collect_site_means_valid_is_the_masked_average(self):
        from eval.floors import collect_site_means
        from sae.dense import sparse_topk_to_dense
        inf, bank, _, pt, *_ = _task_setup()
        site = (0, "attn")
        valid = torch.ones(pt.shape, dtype=torch.bool)
        valid[:, 5:] = False
        valid[3, 2:] = False
        got = collect_site_means(inf, bank, pt, {site}, valid=valid)[site]
        dense = []

        def hook(layer_idx, acts):
            if layer_idx == 0:
                ta, ti = bank.encode(acts[0], "attn", 0)
                dense.append(sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=torch.float32))

        inf.forward(pt, activations_callback=hook)
        want = dense[0][valid].mean(0)
        assert torch.allclose(got, want, atol=1e-6)
        assert torch.allclose(collect_site_means(inf, bank, pt, {site})[site], dense[0].mean((0, 1)), atol=1e-6)
        with pytest.raises(ValueError, match="no position"):
            collect_site_means(inf, bank, pt, {site}, valid=torch.zeros_like(valid))

    def test_precomputed_means_are_used_and_default_is_unchanged(self, monkeypatch):
        """means_A / means_C (081, 2026-10-21): the same values give the same row without collecting anything; other
        values move only the A / C frames; a missing site is refused."""
        from eval.floors import collect_site_means
        inf, bank, alphas, pt, pa, cf, good, bad = _task_setup()
        m = {"task": ("logit_diff", good, bad), "obj": ("set_logmass", good[:, None], bad[:, None])}
        base = W.score_task(inf, bank, alphas, pt, pa, 15, cf[:12], m)
        allsites = W.scope_sites(bank, 0, bank.kinds[0], "all")
        mA = collect_site_means(inf, bank, pt[:15], allsites)
        mC = collect_site_means(inf, bank, cf[:12], allsites)
        monkeypatch.setattr(W, "collect_site_means", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
        got = W.score_task(inf, bank, alphas, pt, pa, 15, cf[:12], m, means_A=mA, means_C=mC)
        assert {k: v for k, v in got.items() if k != "secs"} == {k: v for k, v in base.items() if k != "secs"}
        swapped = W.score_task(inf, bank, alphas, pt, pa, 15, cf[:12], m, means_A=mC, means_C=mA)
        assert swapped["task_eM_topk"] == pytest.approx(base["task_eN_topk"], rel=1e-6)
        assert swapped["task_free0"] == base["task_free0"]                               # Z has no means
        with pytest.raises(ValueError, match="means_A lacks"):
            W.score_task(inf, bank, alphas, pt, pa, 15, cf[:12], m, means_A={}, means_C=mC)

    def test_split_and_scope_guards(self):
        inf, bank, alphas, pt, pa, cf, good, bad = _task_setup()
        m = {"task": ("logit_diff", good, bad)}
        with pytest.raises(ValueError, match="held-out"):
            W.score_task(inf, bank, alphas, pt, pa, 20, cf, m)
        with pytest.raises(ValueError, match="counterfactual"):
            W.score_task(inf, bank, alphas, pt, pa, 15, cf[:0], m)
        up = set(W.upstream_sites(bank, *SEED_SITE))
        r = W.score_task(inf, bank, alphas, pt, pa, 15, cf[:12], m, sites=up)
        assert r["n_sites_scope"] == 5 and r["all_sites_ablated"] is True

    def test_pass_rule(self):
        row = {"task_free0": 0.8, "task_freeM_topk": 1.5, "task_freeN_topk": 1.0, "task_nec": 0.9}
        assert W.passes_task(row)
        assert not W.passes_task(dict(row, task_freeN_topk=0.749))
        assert not W.passes_task(dict(row, task_nec=None))
        assert not W.passes_task(dict(row, task_freeM_topk=float("nan")))
        assert W.passes_task({k.replace("task", "obj"): v for k, v in row.items()}, "obj")


class TestTaskScorerSeq:
    """Multi-token answers (seq_diff, 2026-10-21): teacher-forced, summed log-probs over (good, bad) row pairs."""

    def test_answer_logprob_is_the_sum_of_per_token_reads(self):
        g = torch.Generator().manual_seed(3)
        lg = torch.randn(2, 6, 10, generator=g)
        anchors = torch.tensor([1, 2])
        answers = torch.tensor([[4, 7, 2], [5, 1, -1]])
        lp = torch.log_softmax(lg, -1)
        want = [float(lp[0, 1, 4] + lp[0, 2, 7] + lp[0, 3, 2]), float(lp[1, 2, 5] + lp[1, 3, 1])]
        assert W.answer_logprob(lg, anchors, answers).tolist() == pytest.approx(want, rel=1e-6)
        assert W.pair_diff(torch.tensor([3.0, 1.0, 0.5, 2.0])).tolist() == [2.0, -1.5]
        with pytest.raises(ValueError, match="pairs"):
            W.pair_diff(torch.tensor([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="teacher-forced"):
            W.task_metric(lp[:, 0], "seq_diff", answers)

    def test_single_token_answers_reproduce_the_token_metric(self):
        """Every prompt duplicated into a (good, bad) pair with one-token answers: seq_diff equals logit_diff exactly,
        under every frame."""
        inf, bank, alphas, pt, pa, cf, good, bad = _task_setup()
        tok = W.score_task(inf, bank, alphas, pt, pa, 15, cf[:12], {"task": ("logit_diff", good, bad)})
        rows = pt.repeat_interleave(2, 0)
        anc = pa.repeat_interleave(2)
        ans = torch.stack([good, bad], 1).reshape(-1, 1)
        seq = W.score_task(inf, bank, alphas, rows, anc, 30, cf[:12], {"task": ("seq_diff", ans, None)})
        for f in ("free0", "freeM_topk", "freeN_topk", "free0_a1", "nec", "full", "e0", "acc_free0", "acc_full"):
            assert seq["task_" + f] == pytest.approx(tok["task_" + f], rel=1e-5, abs=1e-6), f
        assert seq["task_n_items"] == 5 and seq["task_top1_full"] is None

    def test_multi_token_answers_respect_padding(self):
        """A two-token answer pair: appending padding columns after every row (and -1 slots after the answers)
        changes nothing; the answer positions are anchor and anchor + 1."""
        inf, bank, alphas, pt, pa, cf, good, bad = _task_setup()
        pa = pa.clamp(max=pt.shape[1] - 2)                        # room for a two-token answer
        rows = pt.repeat_interleave(2, 0).clone()
        anc = pa.repeat_interleave(2)
        ans = torch.stack([torch.stack([good, (good + 1) % 64], 1), torch.stack([bad, (bad + 3) % 64], 1)],
                          1).reshape(-1, 2)
        rows[torch.arange(40), anc + 1] = ans[:, 0]                 # teacher forcing: the first answer token
        base = W.score_task(inf, bank, alphas, rows, anc, 30, cf[:12], {"task": ("seq_diff", ans, None)})
        padded = torch.cat([rows, torch.zeros(40, 3, dtype=rows.dtype)], 1)
        ans_p = torch.cat([ans, torch.full((40, 2), -1)], 1)
        v = torch.zeros(padded.shape, dtype=torch.bool)
        v[:, :rows.shape[1]] = True
        cfp = torch.cat([cf[:12], torch.zeros(12, 3, dtype=cf.dtype)], 1)
        got = W.score_task(inf, bank, alphas, padded, anc, 30, cfp, {"task": ("seq_diff", ans_p, None)}, valid=v,
                           cf_valid=v[:12])
        for f in ("free0", "freeM_topk", "freeN_topk", "nec", "full"):
            assert got["task_" + f] == pytest.approx(base["task_" + f], rel=1e-5), f
        with pytest.raises(ValueError, match="even"):
            W.score_task(inf, bank, alphas, rows, anc, 29, cf[:12], {"task": ("seq_diff", ans, None)})
