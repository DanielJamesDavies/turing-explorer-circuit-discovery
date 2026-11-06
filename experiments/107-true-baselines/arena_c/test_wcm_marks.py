"""Arena C (DAN-194) adapter tests. CPU (the model and dictionaries are loaded on the CPU; the SFC reference runs in
~/venvs/sfc107 when it exists). Run in WSL from the repo root:

  HF_HUB_OFFLINE=1 ./.venv/bin/python -m pytest experiments/107-true-baselines/arena_c/test_wcm_marks.py -q -s

Each test prints one "RESULT ..." line with its measured number (the report quotes them).
  1. encode / decode parity with dictionary_learning.AutoEncoder (third_party, read-only; loaded by path), random and
     real activations, every one of the 19 dictionaries (max abs diff)
  2. splice exactness: unpatched, open-gate WCM patcher over all 19 sites, and SFC's decode(f) + error splice at every
     site, against the raw HF model (max abs logit diff)
  3. SFC's logit-difference metric on SVA examples: ours vs SFC's own code (sfc_metric_ref.py in the SFC env); and our
     load_examples selection vs theirs (same prefixes in the same order)
  4. a tiny WCM v3 logit-objective fit (a few steps) end to end, writing a circuit
"""
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wcm_marks as WM  # noqa: E402

SFC_PY = Path.home() / "venvs" / "sfc107" / "bin" / "python"
SVA_PROMPTS = ["The friends", "The teachers near the desk", "The fathers that the chefs",
               "The friends that the dancer visits", "The lawyer that the guards hate"]


def _autoencoder_cls():
    """dictionary_learning.dictionary.AutoEncoder @61ac634, loaded from third_party by path. Our venv lacks einops,
    which dictionary.py imports at the top but AutoEncoder never uses: a stub module stands in for it (test-only)."""
    if "einops" not in sys.modules:
        try:
            import einops  # noqa: F401
        except ImportError:
            sys.modules["einops"] = types.ModuleType("einops")
    p = WM.FC / "dictionary_learning" / "dictionary_learning" / "dictionary.py"
    spec = importlib.util.spec_from_file_location("_dl_dictionary_ref", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AutoEncoder


@pytest.fixture(scope="module")
def mb():
    """(inference, bank) on the CPU; the bank lazy (one dictionary resident at a time where tests release them)."""
    WM.register()
    from config import config
    from model.inference import Inference
    config.model.family = WM.FAMILY
    config.model.dtype = None
    inf = Inference(torch.device("cpu"), family=WM.FAMILY)
    bank = WM.MarksSAEBank(device=torch.device("cpu"), family=WM.FAMILY, preload=False, embed=True)
    return inf, bank


def _tokens(prompts):
    tok = WM.tokenizer()
    ids = [tok(p).input_ids for p in prompts]
    T = max(len(i) for i in ids)
    return torch.tensor([i + [0] * (T - len(i)) for i in ids]), [len(i) for i in ids]


def _site_acts(inf, prompts):
    """{site: [B, T, 512]} activations at every one of the 24 slots, through our capture protocol."""
    toks, _ = _tokens(prompts)
    acts = {}

    def cb(layer, a):
        for k, x in zip(WM.KINDS, a):
            acts[(layer, k)] = x.detach().clone()

    inf.forward(toks, activations_callback=cb, return_activations=False, tokenize_final=False)
    return acts, toks


# ------------------------------------------------------------------------------------------------------------- 1
def test_encode_decode_parity(mb):
    inf, bank = mb
    AE = _autoencoder_cls()
    acts, _ = _site_acts(inf, SVA_PROMPTS)
    worst = dict(rand_f=0.0, rand_xhat=0.0, real_f=0.0, real_xhat=0.0, real_err=0.0)
    g = torch.Generator().manual_seed(0)
    for (l, k) in WM.REAL_SITES:
        # = AE.from_pretrained(path) body (dictionary.py:127-145), with map_location="cpu" added to its t.load (the
        # files hold CUDA tensors; this test runs without a GPU)
        sd = torch.load(str(WM.dict_path(k, l)), map_location="cpu")
        ref = AE(sd["encoder.weight"].shape[1], sd["encoder.weight"].shape[0])
        ref.load_state_dict(sd)
        ref.normalize_decoder()
        x_real = acts[(l, k)].reshape(-1, 512)
        x_rand = torch.randn(64, 512, generator=g) * x_real.std() + x_real.mean(0)
        for tag, x in (("rand", x_rand), ("real", x_real)):
            with torch.no_grad():
                ta, ti = bank.encode(x, k, l)
                f = torch.zeros(x.shape[0], WM.D_SAE).scatter_add_(-1, ti.long(), ta)
                xh = bank.decode(f, k, l)
                f_ref = ref.encode(x)
                xh_ref = ref.decode(f_ref)
            worst[tag + "_f"] = max(worst[tag + "_f"], float((f - f_ref).abs().max()))
            worst[tag + "_xhat"] = max(worst[tag + "_xhat"], float((xh - xh_ref).abs().max()))
            if tag == "real":
                worst["real_err"] = max(worst["real_err"], float(((x - xh) - (x - ref(x))).abs().max()))
        assert bank.normalised[(k, l)] is False                 # their normalize_decoder() is a no-op on these files
        bank.release(k, l)
        del ref
    # the inert slots: zero code, zero decode
    nz = bank.saes["embed"][3]
    assert isinstance(nz, WM.NullSAE) and float(nz.encode(acts[(3, "embed")])[0].abs().max()) == 0.0
    print("RESULT parity (19 dictionaries; max abs diff vs AutoEncoder) %s" % json.dumps(worst))
    assert worst["rand_f"] < 1e-4 and worst["real_f"] < 1e-4 and worst["real_xhat"] < 1e-4


# ------------------------------------------------------------------------------------------------------------- 2
def test_splice_exact(mb):
    inf, bank = mb
    from circuit.instrument.learned_mask import LearnedMaskPatcher
    toks, _ = _tokens(SVA_PROMPTS)
    with torch.no_grad():
        ref = inf.model.hf(input_ids=toks).logits                       # the raw HF model, no hooks
        _, ours, _ = inf.forward(toks, all_logits=True, return_activations=False, tokenize_final=False)
        sites = list(WM.REAL_SITES)
        thetas = {s: torch.full((WM.D_SAE,), 40.0) for s in sites}     # sigmoid(40) == 1.0 in float32: open gates
        w, b = bank.preact_vector("resid", 5, 0)
        p = LearnedMaskPatcher(bank, thetas, 5, "resid", w, b, tap_seed=False)
        _, lm, _ = inf.forward(toks, patcher=p, all_logits=True, return_activations=False, tokenize_final=False)
        calls = []

        class SFCSplice:                                               # SFC's form: decode(f) + (x - decode(f))
            def __call__(self, model):
                from model.hooks import multi_patch
                return multi_patch(model, self.t)

            def t(self, l, k, x):
                calls.append((l, k))
                if (l, k) not in WM.REAL_SITES:
                    return x
                f, err = bank.full_encode(x, k, l)
                return bank.decode(f, k, l) + err

        _, sp, _ = inf.forward(toks, patcher=SFCSplice(), all_logits=True, return_activations=False,
                               tokenize_final=False)
    d_plain = float((ours - ref).abs().max())
    d_mask = float((lm - ref).abs().max())
    d_sfc = float((sp - ref).abs().max())
    print("RESULT splice max|dlogit| vs raw HF: unpatched %.3g | WCM patcher, all 19 sites open %.3g | SFC-style "
          "decode(f)+err at all 19 sites %.3g (logit scale %.1f)" % (d_plain, d_mask, d_sfc, float(ref.abs().max())))
    assert len(calls) == 24 and calls[:4] == [(0, "embed"), (0, "attn"), (0, "mlp"), (0, "resid")]
    assert d_plain == 0.0 and d_mask == 0.0
    # SFC's own splice recomputes x as decode(f) + (x - decode(f)): float32 cancellation on Pythia's ~1e3 logits
    assert d_sfc / float(ref.abs().max()) < 1e-5


# ------------------------------------------------------------------------------------------------------------- 3
def test_metric_matches_sfc(mb):
    inf, _ = mb
    out = {}
    for task in ("simple", "rc"):
        T = WM.build_task(task, "sfc")
        held = [T["rows"][i] for i in T["split"]["held"]][:8]
        toks = torch.tensor([r["tokens"] for r in held])               # min length: no padding
        with torch.no_grad():
            _, lg, _ = inf.forward(toks, all_logits=True, return_activations=False, tokenize_final=False)
        last = lg[:, -1].float()
        ours = (last.gather(1, torch.tensor([[r["good"][0]] for r in held]))
                - last.gather(1, torch.tensor([[r["bad"][0]] for r in held]))).squeeze(1)
        rec = dict(ours=[round(float(x), 5) for x in ours], ref=None)
        if SFC_PY.exists():
            p = subprocess.run([str(SFC_PY), str(HERE / "sfc_metric_ref.py"), "--task", task, "--n", "8"],
                               capture_output=True, text=True, timeout=900,
                               env=dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", HF_HUB_OFFLINE="1"))
            line = [x for x in p.stdout.splitlines() if x.startswith("JSON")]
            assert line, p.stderr[-3000:]
            r = json.loads(line[0][4:])
            disc_ours = [T["rows"][i]["prompt"] for i in T["split"]["train"]]
            ev_ours = [T["rows"][i]["prompt"] for i in T["split"]["held"]]
            assert disc_ours == r["discovery_prefixes"], "discovery selection differs from SFC's load_examples"
            assert ev_ours == r["eval_prefixes"], "evaluation selection differs from SFC's load_examples"
            assert [[x["meta"]["clean_answer"], x["meta"]["patch_answer"]] for x in held] == r["eval_answers"][:8]
            rec["ref"] = r["metric"]
            rec["max_abs_diff"] = float((ours - torch.tensor(r["metric"])).abs().max())
            rec["source"] = "SFC's own code (sfc107: nnsight %s)" % "0.3.7"
        else:
            rec["source"] = "hand computed only (no ~/venvs/sfc107)"
        out[task] = rec
    print("RESULT metric %s" % json.dumps({k: dict(max_abs_diff=v.get("max_abs_diff"), source=v["source"],
                                                    ours_first3=v["ours"][:3], ref_first3=(v["ref"] or [])[:3])
                                           for k, v in out.items()}))
    for v in out.values():
        if v["ref"] is not None:
            assert v["max_abs_diff"] < 1e-3


# ------------------------------------------------------------------------------------------------------------- 4
def test_tiny_fit_writes_circuit(tmp_path):
    os.environ.pop("STOP_AT", None)
    eng = WM.make_engine("noembed", device="cpu", means="per_position")
    rows = WM.run(["simple"], [1e-3], mode="sfc", steps=6, out=tmp_path, eng=eng)
    M = eng.means(eng.batch(WM.build_task("simple", "sfc")))
    assert set(M["A"]) == set(WM.scope_sites("noembed")) and tuple(M["A"][(2, "resid")].shape) == (2, WM.D_SAE)
    assert rows[0]["means"] == "per_position" and rows[0]["sites"] == "noembed"
    assert len(rows) == 1
    r = rows[0]
    circ = tmp_path / r["circuit"]
    assert circ.exists()
    c = json.loads(circ.read_text())
    assert c["family"] == WM.FAMILY and c["wcm"] == WM.V3_LOGIT
    f, cv = c["fit"], c["curve"]
    terms = {k: v for k, v in cv.items() if k.startswith("terms.")}
    print("RESULT tiny fit (simple, 6 steps, CPU, behavioural v3: Z + A + C + R): %d latents, task Z %s A %s C %s nec %s, fit "
          "%.1fs, score %.1fs | sparse_fill_calls %s, resample_weight %s, donors %s, draws %s, resample_fill_calls %s, "
          "rank_weight %s (%s) | curve rank %s | terms %s" % (
              r["n_latents"], r.get("task_free0"), r.get("task_freeM_topk"), r.get("task_freeN_topk"),
              r.get("task_nec"), r["t_fit"], r["t_score"], f.get("sparse_fill_calls"), f.get("resample_weight"),
              f.get("resample_n_donors"), f.get("resample_draws"), f.get("resample_fill_calls"), f.get("rank_weight"),
              f.get("rank_mode"), cv.get("rank"), json.dumps(terms)))
    assert r["n_latents"] > 0 and r["task_full"] is not None
    assert "embed" not in r["latents_per_site"]
    assert f.get("sparse_fill_calls", 0) > 0                          # v3's sparsity-preserving fill ran
    assert f.get("resample_weight") == 1.0 and f.get("resample_draws", 0) > 0 and f.get("resample_n_donors") == 100
    assert not f.get("rank_weight")                                   # behavioural WCM has no rank-keep (2026-11-02)
