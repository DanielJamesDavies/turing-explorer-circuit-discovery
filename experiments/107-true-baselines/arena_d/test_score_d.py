"""CPU tests for score_d.py (arena D4) on the tiny random Gemma-2 of test_wcm_gemma.py (vocab 256k, the real rc prompts),
both ways: ours (repo .venv) and theirs (caples107 subprocess, their helpers/utils.py unmodified).

  cd <repo> && CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 ./.venv/bin/python -m pytest \
      experiments/107-true-baselines/arena_d/test_score_d.py -q -p no:cacheprovider -s
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_wcm_gemma as TW  # noqa: E402
import wcm_gemma as W  # noqa: E402
import score_d as S  # noqa: E402

torch.set_num_threads(2)
FAM = TW.TINY_RES
NB, BS, T, MB = 10, 16, 7, TW.CAPLES_TINY["mean_batches"]          # their data: 160 records = 10 batches of 16


def _fake_caples_run(path: Path, layers, seed: int = 3) -> Path:
    """A run folder as do_training_run writes it (--per_token_mask): sae_mask_{i}.pt = SparseMask.state_dict() and the
    run JSON (per hook: torch.where(theta > 0)[1]; total_density). One BOS-row member at the first SAE."""
    path.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    d = {}
    tot = 0
    for i, l in enumerate(layers):
        th = torch.randn(T, TW.DS, generator=g) - 0.9                 # ~18% positive
        th[0] = -1.0
        if i == 0:
            th[0, 5] = 0.4                                            # a BOS pair: counted by them, no effect
        torch.save({"mask": th}, path / ("sae_mask_%d.pt" % i))
        d["blocks.%d.hook_resid_post" % l] = torch.where(th > 0)[1].tolist()
        tot += int((th > 0).sum())
    d.update(total_density=tot, avg_density=tot / len(layers), faithfulness=None, logit_diff=None)
    (path / "0.0224_run.json").write_text(json.dumps(d))
    return path


@pytest.fixture(scope="module")
def D(tmp_path_factory):
    if not TW.CAPLES_PY.exists():
        pytest.skip("caples107 venv not found")
    os.environ.pop("STOP_AT", None)
    os.environ.pop("STOP_EPOCH", None)
    tmp = tmp_path_factory.mktemp("d4")
    inf, bank = TW.tiny_stack(FAM, vocab=TW.GEMMA_VOCAB)
    tok = W.tokenizer()
    task = W.build_task("rc", "caples", tok=tok, family=FAM, **TW.CAPLES_TINY)
    # their data and model as a caples_means.py fixture: the SAME tiny model and dictionaries as ours
    recs = W._caples_filter(W.CAPLES_SRC / "data" / "sva" / "rc_train.json", tok, T, NB * BS)
    clean = torch.tensor([tok(e["clean_prefix"]).input_ids for e in recs]).reshape(NB, BS, T)
    corr = torch.tensor([tok(e["patch_prefix"]).input_ids for e in recs]).reshape(NB, BS, T)
    cl = torch.tensor([W._answer_id(tok, e["clean_answer"]) for e in recs]).reshape(NB, BS)
    pl = torch.tensor([W._answer_id(tok, e["patch_answer"]) for e in recs]).reshape(NB, BS)
    m = inf.model.hf
    saes = {l: bank.saes["resid"][l] for l in TW.RES_LAYERS}
    fx = dict(hf_config=dict(TW.TINY_CFG, vocab_size=TW.GEMMA_VOCAB),
              hf_state={k: v.clone() for k, v in m.state_dict().items()},
              saes={l: dict(W_enc=s.encoder.weight.detach().T.contiguous(), W_dec=s.decoder.weight.detach().T.contiguous(),
                            b_enc=s.encoder.bias.detach().clone(), b_dec=s.decoder_bias.detach().clone(),
                            threshold=s.threshold.clone()) for l, s in saes.items()},
              clean=clean, corr=corr, clean_lab=cl, corr_lab=pl, tokenizer=W.HF_ID)
    torch.save(fx, tmp / "fixture.pt")
    means = tmp / "means.pt"
    env = dict(os.environ, WANDB_MODE="disabled", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2")
    p = subprocess.run([str(TW.CAPLES_PY), "-I", str(HERE / "caples_means.py"), "--fixture", str(tmp / "fixture.pt"),
                        "--mean-batches", str(MB), "--batch-size", str(BS), "--example-length", str(T),
                        "--device", "cpu", "--out", str(means)], capture_output=True, text=True, env=env, timeout=1800)
    assert p.returncode == 0, p.stderr[-4000:]
    engines = {md: W.make_engine(FAM, md, "per_position", str(means) if md == "caples" else None, shared=(inf, bank))
               for md in ("native", "caples")}
    # one WCM circuit: a real 6-step fit in Caples mode (its eval row = score_task's numbers)
    fit = W.run(["rc"], [1e-4], family=FAM, data="caples", mode="caples", means="per_position", steps=6,
                out=tmp / "fit", eng=engines["caples"], log=lambda s: None, caples_means=str(means),
                task_kw=TW.CAPLES_TINY, tok=tok)
    assert len(fit) == 1 and fit[0]["n_latents"] > 0
    layers = [l for l, _ in bank.real_sites]
    cap = S.caples_circuit(_fake_caples_run(tmp / "caples_run", layers), layers)
    wcm = S.wcm_circuit(tmp / "fit" / fit[0]["circuit"], bank.d_sae, name="wcm")
    # a sparse WCM-format circuit (every 3rd member of the fit, its alphas kept): the tiny fit keeps nearly everything
    rec = json.loads((tmp / "fit" / fit[0]["circuit"]).read_text())
    rec["alphas"] = {s: {i: a for j, (i, a) in enumerate(sorted(d.items(), key=lambda t: int(t[0]))) if j % 3 == 0}
                     for s, d in rec["alphas"].items()}
    (tmp / "wcm_sub.json").write_text(json.dumps(rec))
    wcm_sub = S.wcm_circuit(tmp / "wcm_sub.json", bank.d_sae, name="wcm_sub")
    wcm_b = dict(wcm_sub, name="wcm_broadcast", keep={l: k.expand(T, -1).clone() for l, k in wcm_sub["keep"].items()},
                 scale={l: v.expand(T, -1).clone() for l, v in wcm_sub["scale"].items()})
    circuits = [cap, wcm, wcm_sub, wcm_b, S.keep_all_circuit(layers, T, bank.d_sae), S.empty_circuit()]
    ev = S.their_batches("rc", tok, NB * BS, BS, T)
    out = tmp / "score"
    rows = S.run_ours(engines, task, circuits, out, their_eval=ev, log=lambda s: None)
    S.write_masks(circuits, T, out / "masks.pt")
    p = subprocess.run([str(TW.CAPLES_PY), "-I", str(HERE / "score_d.py"), "theirs", "--fixture", str(tmp / "fixture.pt"),
                        "--masks", str(out / "masks.pt"), "--caples-means", str(means), "--out", str(out / "theirs.json"),
                        "--batch-size", str(BS), "--example-length", str(T), "--device", "cpu"],
                       capture_output=True, text=True, env=env, timeout=1800)
    assert p.returncode == 0, p.stderr[-4000:]
    theirs = json.loads((out / "theirs.json").read_text())
    merged = S.merge(rows, theirs, out)
    R = {(r["circuit"], r["substrate"]): r for r in rows}
    return dict(tmp=tmp, out=out, engines=engines, task=task, fit=fit[0], wcm=wcm, cap=cap, rows=R, theirs=theirs,
                merged=merged, means=means, inf=inf, bank=bank)


SUBS = ("native", "caples_theirs")
FRAMES = ("full", "e0", "eM_topk", "eN_topk", "free0_raw", "freeM_topk_raw", "freeN_topk_raw", "nec_raw")


@pytest.mark.parametrize("sub", SUBS)
def test_keep_all_is_clean_ld_exactly(D, sub):
    r = D["rows"][("keep_all", sub)]
    for f in ("free0_raw", "freeM_topk_raw", "freeN_topk_raw"):
        assert r["task_" + f] == r["task_full"], (sub, f, r["task_" + f], r["task_full"])
    for f, e in (("free0", "e0"), ("freeM_topk", "eM_topk"), ("freeN_topk", "eN_topk")):
        # the ratio is 1 exactly, or None when the random model's empty circuit is not below full (score_task's rule)
        assert r["task_" + f] == (1.0 if r["task_" + e] < r["task_full"] else None), (sub, f)
    e = D["rows"][("empty", sub)]
    assert e["task_free0_raw"] == e["task_e0"] and e["task_nec_raw"] == e["task_full"]
    print("RESULT keep-all %s: LD %.6f = full %.6f (Z / A / C raw equal, exact)" % (sub, r["task_free0_raw"],
                                                                                 r["task_full"]))


def test_collapsed_circuit_matches_score_task(D):
    """Our per-position path on a position-collapsed WCM circuit = eval.wcm_eval.score_task (via 081's Engine.score) in
    both substrates; the same circuit broadcast to [T, d_sae] scores identically."""
    from eval import wcm_eval as WE  # noqa: F401
    import task_common as C
    worst = 0.0
    for md, sub in (("native", "native"), ("caples", "caples_theirs")):
        eng = D["engines"][md]
        S.use_mode(eng)
        B = eng.batch(D["task"])
        for name, path in (("wcm", D["tmp"] / "fit" / D["fit"]["circuit"]), ("wcm_sub", D["tmp"] / "wcm_sub.json")):
            ref = eng.score(B, C.alphas_from_json(json.loads(path.read_text())["alphas"]))
            ours = D["rows"][(name, sub)]
            for f in FRAMES:
                a, b = ours["task_" + f], ref["task_" + f]
                worst = max(worst, abs(a - b))
                assert abs(a - b) <= 1e-4 * (1 + abs(b)), (name, sub, f, a, b)
        ours = D["rows"][("wcm_sub", sub)]
        bc = D["rows"][("wcm_broadcast", sub)]
        for f in FRAMES + ("nec_errclean_raw",):
            assert abs(bc["task_" + f] - ours["task_" + f]) <= 1e-6 * (1 + abs(ours["task_" + f])), (sub, f)
    # the fit's own eval row (run(): score_task in Caples mode) agrees too
    for f in FRAMES:
        assert abs(D["rows"][("wcm", "caples_theirs")]["task_" + f] - D["fit"]["task_" + f]) <= 1e-4 * (
            1 + abs(D["fit"]["task_" + f])), f
    print("RESULT collapsed WCM circuits (%d and %d latents) vs score_task: max |diff| %.2e over %d frames x 2 "
          "substrates" % (D["rows"][("wcm", "native")]["n_latents"], D["rows"][("wcm_sub", "native")]["n_latents"],
                          worst, len(FRAMES)))


def test_necessity_two_columns(D):
    for c in ("empty", "wcm", "wcm_sub", "caples:caples_run"):
        n = D["rows"][(c, "native")]
        assert n["task_nec_errclean_raw"] is not None and n["nec_errclean_note"] is None
        k = D["rows"][(c, "caples_theirs")]
        assert k["task_nec_errclean_raw"] == k["task_nec_raw"] and "eps_bar" in k["nec_errclean_note"]
    e = D["rows"][("empty", "native")]
    # no member removed, every error at its clean value: the clean run, up to decode(c) + (x - decode(c)) rounding
    assert abs(e["task_nec_errclean_raw"] - e["task_full"]) <= 1e-4 * (1 + abs(e["task_full"]))
    w = D["rows"][("wcm", "native")]
    print("RESULT necessity (native) wcm: live %s, error clean %s; caples mask: live %s, error clean %s" % (
        S._f(w["task_nec"]), S._f(w["task_nec_errclean"]), S._f(D["rows"][("caples:caples_run", "native")]["task_nec"]),
        S._f(D["rows"][("caples:caples_run", "native")]["task_nec_errclean"])))


def test_caples_membership_and_sizes(D):
    cap, r = D["cap"], D["rows"][("caples:caples_run", "native")]
    assert cap["meta"]["json_checked"]
    js = json.loads((D["tmp"] / "caples_run" / "0.0224_run.json").read_text())
    assert r["pairs_all"] == js["total_density"] and r["n_bos_pairs"] == 1
    assert r["pairs_nonbos"] == r["pairs_all"] - 1
    assert r["n_latents"] == sum(int(k[1:].any(0).sum()) for k in cap["keep"].values())
    w = D["rows"][("wcm", "native")]
    assert w["n_latents"] == D["fit"]["n_latents"] and w["pairs_nonbos"] == D["fit"]["native_nodes"]
    assert w["pairs_all"] == w["n_latents"] * T and w["n_bos_pairs"] == w["n_latents"]
    bad = D["tmp"] / "caples_bad"
    _fake_caples_run(bad, [l for l, _ in D["bank"].real_sites])
    js_bad = json.loads((bad / "0.0224_run.json").read_text())
    js_bad["total_density"] += 1
    (bad / "0.0224_run.json").write_text(json.dumps(js_bad))
    with pytest.raises(ValueError, match="total_density"):
        S.caples_circuit(bad, [l for l, _ in D["bank"].real_sites])


def test_theirs_both_ways(D):
    """Their SAEMasks evaluation (caples107) of one Caples mask and one WCM circuit, and its re-statement in our HF
    stack (their_protocol) on the same batches and substrate, agree; their references reproduce caples_means.py's."""
    th = D["theirs"]
    refs = th["refs"]
    fm = torch.load(D["means"], map_location="cpu", weights_only=False)
    ev = fm["ld"]["eval"]
    assert th["meta"]["eval_batches"] == [(-i) % NB for i in range(10)] == fm["meta"]["eval_batches"]["eval"]
    assert abs(refs["ld_model"] - sum(ev["model"]) / 10) <= 1e-5 * (1 + abs(refs["ld_model"]))
    assert abs(refs["ld_spliced"] - sum(ev["sae_mean_error"]) / 10) <= 1e-5 * (1 + abs(refs["ld_spliced"]))
    ka = th["circuits"]["keep_all"]
    assert abs(ka["ld"] - refs["ld_spliced"]) <= 1e-5 * (1 + abs(refs["ld_spliced"]))   # mean + (c - mean) * 1
    assert th["circuits"]["empty"]["ld"] == refs["ld_empty"]
    lines = []
    for name in ("caples:caples_run", "wcm", "wcm_sub", "keep_all", "empty"):
        t = th["circuits"][name]
        o = D["rows"][(name, "caples_theirs")]["their_protocol"]
        for k in ("ld", "ld_empty", "ld_spliced", "ld_model"):
            tv = t[k] if k == "ld" else refs[k]
            assert abs(o[k] - tv) <= 1e-4 * (1 + abs(tv)), (name, k, o[k], tv)
        for k in ("f_model", "sfc_model", "f_spliced", "sfc_spliced"):
            if t[k] is not None:
                assert abs(o[k] - t[k]) <= 1e-3 * (1 + abs(t[k])), (name, k, o[k], t[k])
        lines.append("%s: theirs ld %.4f f %.3f sfc %.3f (pairs %d) | ours on their batches ld %.4f" % (
            name, t["ld"], t["f_model"], t["sfc_model"] if t["sfc_model"] is not None else float("nan"), t["n_pairs"],
            o["ld"]))
    assert th["circuits"]["caples:caples_run"]["n_pairs"] == D["rows"][("caples:caples_run", "native")]["pairs_all"]
    m = {r["circuit"]: r for r in D["merged"]}
    assert set(m["wcm"]["ours"]) == {"native/per_position", "caples_theirs/per_position"} and "theirs" in m["wcm"]
    print("RESULT theirs (refs model %.4f spliced %.4f empty %.4f):\n  %s" % (
        refs["ld_model"], refs["ld_spliced"], refs["ld_empty"], "\n  ".join(lines)))
