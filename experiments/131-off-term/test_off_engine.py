"""131 CPU tests on one real GPT-2 small target (production record, a few steps):
  (a) the patched engine at w_O = 0 is bit-identical to the unpatched src engine (same circuit: members and
      coefficients; same loss curve and per-term curves)
  (b) at w_O = 1 the off term is built, appears in the training curve, is nonzero, and its gradient reaches the gates
      (theta) and the coefficients (psi)
  (c) the off term only ever reads the 48 contrast-TRAINING rows: its tokens are eval_reference's C rows (= the R
      donors = rec["neg"][:48]), none equals a held-out contrast row, every row index it uses is < 48, and no donor
      draw is the row itself
  (d) the magnitude read: on the verified-silent held-out contrast contexts the FULL MODEL's post-Top-K target is 0 at
      every position (natural_silent_max = 0), and the capture reproduces the donor test's own silent_firing
  (+) the patched source applies every anchor once

  wsl: CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python -X utf8 experiments/131-off-term/test_off_engine.py
  env: KEY [1.mlp.22405]  STEPS [4]
Writes only under experiments/131-off-term/out_cputest/ (a fresh run dir per call).
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "out_cputest" / time.strftime("%Y%m%d_%H%M%S")
os.environ["OUT131"] = str(RUN_DIR)
os.environ.setdefault("STEPS", "4")
os.environ["DONOR_DRAWS"] = "1"
os.environ.setdefault("SAE_PRELOAD", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
KEY = os.environ.get("KEY", "1.mlp.22405")
sys.path.insert(0, str(HERE))
import run as R131  # noqa: E402

R131.setup_env()
import torch  # noqa: E402
import circuit.instrument.learned_mask as ORIG  # noqa: E402,F401  (the unpatched src module, imported first)
import driver  # noqa: E402

RESULTS = {}


def sig(c):
    """Members and coefficients (node keys are per-circuit uuids: compared by feature id)."""
    return sorted((str(n.metadata.get("feature_id")), str(n.metadata.get("role")), float(n.metadata.get("amplitude", 1.0)))
                  for n in c.nodes.values())


def fit(R, M, rec, tag):
    tr_arm, _ = R.train_arm(rec)
    c, t = R.fit(M, rec, KEY, tr_arm, RUN_DIR / "circuits" / tag / ("%s.pt" % KEY))
    curve = (M.last_mask_provenance or {}).get("curve") or {}
    return c, curve, t


def main():
    assert driver.STEPS == int(os.environ["STEPS"])
    import off_engine
    src = off_engine.patched_source()
    for _a, repl in off_engine.EDITS:
        assert repl in src
    print("(+) every anchor applied once; patched source %d chars" % len(src), flush=True)
    R = driver.Runner()
    assert str(R.dev) == "cpu", R.dev
    M = R.method(driver.GAMMA, driver.LAM, True)
    rec = R.contexts(KEY)
    t0 = time.time()
    c0, cur0, t_a = fit(R, M, rec, "unpatched")
    print("unpatched fit: %d steps, n %d, %.0fs, terms %s" % (driver.STEPS, driver.n_nodes(c0), t_a,
                                                               list(cur0.get("terms", {}))), flush=True)
    mod = off_engine.install()
    assert sys.modules["circuit.instrument.learned_mask"] is mod
    off_engine.configure(0.0)
    c1, cur1, _ = fit(R, M, rec, "w0")
    same_c = sig(c0) == sig(c1)
    same_loss = cur0["loss"] == cur1["loss"]
    same_terms = cur0.get("terms") == cur1.get("terms")
    assert off_engine.STATS["built"] == 0 and "off" not in cur1.get("terms", {})
    RESULTS["a"] = dict(identical_circuit=same_c, identical_loss_curve=same_loss, identical_term_curves=same_terms,
                        n=driver.n_nodes(c1), loss_unpatched=cur0["loss"], loss_w0=cur1["loss"])
    print("(a) w_O = 0 vs unpatched: circuit identical %s | loss curve identical %s | term curves identical %s"
          % (same_c, same_loss, same_terms), flush=True)
    assert same_c and same_loss and same_terms

    # (b) + (c): w_O = 1
    off_engine.configure(1.0)
    seen = {}
    ref_c = None

    def on_make(p):
        from circuit.protocol_contexts import eval_reference
        nonlocal ref_c
        ref_c = eval_reference(rec, R.dev, split="explicit")[1]
        seen["tokens_eq_eval_ref_C"] = bool(torch.equal(p.off_tokens.cpu(), ref_c.cpu()))
        seen["floor_shape"] = list(p.off_floor.shape)

    def on_loss(pat, rows, ex):
        if "grad" in seen:
            return
        th = list(pat.thetas.values())
        am = list(pat.amps.values())
        g = torch.autograd.grad((ex ** 2).mean(), th + am, retain_graph=True, allow_unused=True)
        gt = [x for x in g[:len(th)] if x is not None]
        ga = [x for x in g[len(th):] if x is not None]
        seen["grad"] = dict(theta_norm=float(torch.stack([x.norm() for x in gt]).norm()) if gt else 0.0,
                            psi_norm=float(torch.stack([x.norm() for x in ga]).norm()) if ga else 0.0,
                            first_loss=float((ex ** 2).mean()), rows=rows.tolist(),
                            donor_rows=pat.donors.rows.tolist())
    off_engine.STATS.update(on_make=on_make, on_loss=on_loss)
    c2, cur2, _ = fit(R, M, rec, "w1")
    st = off_engine.stats_summary()
    off_curve = cur2.get("terms", {}).get("off")
    chk = R131.off_rows_check(rec, R.dev, M)
    RESULTS["b"] = dict(term_names=list(cur2.get("terms", {})), off_curve=off_curve, grad=seen.get("grad"),
                        loss_w1=cur2["loss"], n=driver.n_nodes(c2), stats={k: v for k, v in st.items()
                                                                           if k not in ("on_make", "on_loss")})
    print("(b) w_O = 1: terms %s | off curve %s | grad |theta| %.3g |psi| %.3g | n %d (w0 %d)" % (
        list(cur2.get("terms", {})), [round(v, 6) for v in (off_curve or [])], seen["grad"]["theta_norm"],
        seen["grad"]["psi_norm"], driver.n_nodes(c2), driver.n_nodes(c1)), flush=True)
    assert off_curve is not None and any(v > 0 for v in off_curve), "off term zero or missing"
    assert seen["grad"]["theta_norm"] > 0 and seen["grad"]["psi_norm"] > 0, "no gradient from O"
    RESULTS["c"] = dict(checks=chk, tokens_eq_eval_ref_C=seen.get("tokens_eq_eval_ref_C"), rows_seen=st["rows_seen"],
                        max_row=st["max_row"], self_draws=st["self_draws"], draws=st["draws"], n_off=st["n_off"])
    print("(c) off rows: %s | tokens == eval_reference C %s | rows used %d (max index %s) | self draws %d of %d"
          % (chk, seen.get("tokens_eq_eval_ref_C"), st["rows_seen"], st["max_row"], st["self_draws"], st["draws"]),
          flush=True)
    assert chk["off_eq_resample_donors"] and chk["off_eq_neg48"] and chk["off_rows_in_heldout"] == 0
    assert seen["tokens_eq_eval_ref_C"] and st["max_row"] < st["n_off"] == 48 and st["self_draws"] == 0
    off_engine.configure(0.0)

    # (d) the magnitude read, one donor draw, on the w_O = 1 circuit
    import eval.donor_test as DT
    cap = R131.Capture(DT)
    pool = torch.load(R131.POOL, weights_only=False, map_location="cpu")["tokens"]
    cap.calls, cap.on = [], True
    drows, _ = DT.donor_target(R.G, KEY, rec, {"o1": c2}, pool, variants=R131.VARIANTS, draws=1)
    cap.on = False
    mag = R131.magnitude(cap.calls, drows["o1"])
    RESULTS["d"] = mag
    print("(d) full model on the 16 silent contexts: max %s mean %s | circuit SF_C %s mag mean %s p95 %s | "
          "capture vs row SF diff %s" % (mag.get("natural_silent_max"), mag.get("natural_silent_mean"),
                                         mag["fit_C"]["sf"], mag["fit_C"]["mean"], mag["fit_C"]["p95"],
                                         mag.get("sf_check_max_diff")), flush=True)
    assert mag["natural_silent_max"] == 0.0 and mag["sf_check_max_diff"] == 0.0
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "test_results.json").write_text(json.dumps(RESULTS, indent=1, default=str) + "\n")
    print("ALL OK (%.0fs; %s)" % (time.time() - t0, RUN_DIR / "test_results.json"), flush=True)


if __name__ == "__main__":
    main()
