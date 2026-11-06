"""101: the native resample term (src learned_mask resample_*) against the in-memory patches it replaces, on a CPU fake
model: same scores, same amplitudes, same per-step training curve, bit for bit.

  v2CR    095 rs2_engine, SPEED drop_floor off / sp_donor on, RESAMPLE weight 0.25 (091 setup_terms' factory and
          DonorFloors), train_fill "sparse"      vs  native resample_weight 0.25 under train_fill "sparse"
  v1 + R  079's raw R (rs2_engine, sp_donor off), train_fill "dense"   vs  native under "dense"
  k = 4   100 e4_engine E4 k 4                                          vs  native resample_k 4
  U, dense  unweighted, train_fill "dense": the patch and the native term agree
  U, sparse unweighted, train_fill "sparse": they DIFFER by design (the patch's donor patcher kept the dense donor fill
          on the gate-only branch; native gives it the sparsity-preserving donor fill, as the weighted branch)

  PYTHONPATH=src python -m pytest -q experiments/101-r-native/test_101_equiv_cpu.py
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
sys.path.insert(0, str(REPO / "tests" / "circuit" / "instrument"))
for d in ("079-resample-training", "092-v2-speedups", "095-v2-resample-appendix", "100-scrub-frontier"):
    sys.path.insert(0, str(EXP / d))

import circuit.instrument as PKG  # noqa: E402
from circuit.instrument import learned_mask as LM  # noqa: E402
import rs_engine  # noqa: E402
import rs2_engine  # noqa: E402
import e4_engine  # noqa: E402
import test_learned_mask_resample as T  # noqa: E402  (the fake model of the src unit tests)

W_R = 0.25


def _args(**kw):
    a = dict(objective="pos", sites=[T.SITE], seed_layer=1, seed_kind="resid", seed_latent_idx=3, pos_tokens=T.PT,
             pos_argmax=torch.zeros(8, dtype=torch.long), neg_tokens=T.NT, steps=40, lr=0.1, l1_lambda=1e-3,
             batch_size=4, log_every=0, mask_floor_source="triple", free_amplitude=True, binarize="anneal",
             triple_floor_weight=0.25, dual_floor_weight=0.25, train_fill="sparse", sparse_grad="ste",
             rank_weight=3e-3, rank_mode="keep")
    a.update(kw)
    return a


def _patched(engine, sp_donor, k=1, **kw):
    """One fit through a historical patch: the patched module is installed for the call only."""
    saved = sys.modules[LM.__name__], PKG.learned_mask
    try:
        mod = engine.install()
        engine.reset(mod)
        mod.SPEED.update(drop_floor=False, sp_donor=sp_donor)
        if k > 1:
            mod.E4["k"] = k
        inf, bank = T._model()
        codes = LM.capture_donor_codes(inf, bank, T.DONORS, [T.SITE])           # == 073 capture_codes on this model
        floors = rs_engine.DonorFloors(codes, bank.d_sae, bank.device, seed=LM.RESAMPLE_SEED)
        mod.RESAMPLE.update(weight=W_R, make=rs_engine.make_factory(mod, floors))
        out = mod.run_learned_mask(inf, bank, **_args(**kw))
        engine.reset(mod)
        return out
    finally:
        sys.modules[LM.__name__], PKG.learned_mask = saved


def _native(k=1, **kw):
    inf, bank = T._model()
    return LM.run_learned_mask(inf, bank, **_args(resample_weight=W_R, resample_tokens=T.DONORS, resample_k=k, **kw))


def _same(a, b, names=None):
    (sa, pa), (sb, pb) = a, b
    assert sa == sb
    assert pa["amp_kept"] == pb["amp_kept"]
    assert pa["curve"]["loss"] == pb["curve"]["loss"]
    ta, tb = pa["curve"]["terms"], pb["curve"]["terms"]
    for n in (names or ("zero", "floor", "pos", "resample")):
        assert ta[n] == tb[n], n


def test_v2cr_native_equals_095_patch():
    nat, pat = _native(), _patched(rs2_engine, sp_donor=True)
    _same(nat, pat)
    assert nat[1]["resample_fill_calls"] == 40 and nat[1]["resample_draws"] == 40
    assert nat[1]["sparse_fill_calls"] == pat[1]["sparse_fill_calls"]


def test_dense_r_native_equals_079_patch():
    _same(_native(train_fill="dense"), _patched(rs2_engine, sp_donor=False, train_fill="dense"))


def test_k4_native_equals_100_e4():
    nat, pat = _native(k=4), _patched(e4_engine, sp_donor=True, k=4)
    assert nat[0] == pat[0] and nat[1]["amp_kept"] == pat[1]["amp_kept"]
    assert nat[1]["curve"]["loss"] == pat[1]["curve"]["loss"]
    assert nat[1]["resample_draws"] == 160


def test_unweighted_dense_agrees():
    _same(_native(free_amplitude=False, train_fill="dense"),
          _patched(rs2_engine, sp_donor=True, free_amplitude=False, train_fill="dense"))


def test_unweighted_sparse_differs_by_design():
    nat = _native(free_amplitude=False)
    pat = _patched(rs2_engine, sp_donor=True, free_amplitude=False)
    assert nat[1]["curve"]["terms"]["zero"][0] == pat[1]["curve"]["terms"]["zero"][0]
    assert nat[1]["curve"]["terms"]["resample"] != pat[1]["curve"]["terms"]["resample"]
    assert nat[1]["resample_fill_calls"] == 40


def test_historical_patches_still_build_on_this_src():
    for eng in (rs_engine, rs2_engine, e4_engine):
        compile(eng.patched_source(), "<%s>" % eng.__name__, "exec")
    assert "patcher_donor" in rs2_engine.patched_source()        # the native code is present (dormant) in the patch


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
