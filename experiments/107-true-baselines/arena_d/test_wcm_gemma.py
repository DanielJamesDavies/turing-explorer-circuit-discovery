"""CPU tests for wcm_gemma.py (DAN-203), on a tiny random Gemma-2 (no download) plus the cached real dictionaries.

  cd <repo> && CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 ./.venv/bin/python -m pytest \
      experiments/107-true-baselines/arena_d/test_wcm_gemma.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wcm_gemma as W  # noqa: E402

D, DS, NL = 32, 48, 3
TINY_CFG = dict(hidden_size=D, intermediate_size=64, num_hidden_layers=NL, num_attention_heads=2,
                num_key_value_heads=1, head_dim=16, max_position_embeddings=64, sliding_window=8,
                query_pre_attn_scalar=16)
GEMMA_VOCAB = 256000                              # the real tokenizer's ids (task fits use the real SVA prompts)


def tiny_gemma(seed: int = 0, vocab: int = 128):
    from transformers import Gemma2Config, Gemma2ForCausalLM
    torch.manual_seed(seed)
    cfg = Gemma2Config(vocab_size=vocab, attn_implementation="eager", **TINY_CFG)
    m = Gemma2ForCausalLM(cfg).eval()
    with torch.no_grad():                         # RMSNorm weights start at 0; make (1 + w) non-trivial
        for n, p in m.named_parameters():
            if "norm" in n:
                p.uniform_(-0.5, 0.5)
    return m


def fam(kind: str):
    from model import families as F
    return F.ModelFamily(name="tiny-" + kind, hf_model_id=None, tokenizer_id="none", n_layer=NL, d_model=D,
                         kinds=(kind,), same_layer_upstream={kind: ()}, sae=None)


def rand_state(seed: int):
    """jumprelu_from's arguments: W_enc [DS, D], b_enc, W_dec [DS, D], b_dec, threshold."""
    g = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(DS, D, generator=g) / D ** 0.5
    W_dec = torch.randn(DS, D, generator=g) / DS ** 0.5
    return W_enc, torch.randn(DS, generator=g) * 0.1, W_dec, torch.randn(D, generator=g) * 0.1, \
        torch.rand(DS, generator=g) * 0.3


def rand_sae(seed: int):
    return W.jumprelu_from(*rand_state(seed))


TINY_PLT, TINY_RES = "tiny-gemma-plt", "tiny-gemma-gsres"
RES_LAYERS = (0, 2)


def tiny_families():
    W.register_tiny(TINY_PLT, "tc", NL, D, DS, range(NL), lambda l: rand_state(100 + l))
    W.register_tiny(TINY_RES, "resid", NL, D, DS, RES_LAYERS, lambda l: rand_state(200 + l))
    return TINY_PLT, TINY_RES


def tiny_stack(family: str, vocab: int = 128, seed: int = 0):
    """(inference, bank) on a tiny random Gemma-2 through the arena-D site model and GemmaSAEBank (CPU, fp32)."""
    tiny_families()
    inf = W.make_inference(tiny_gemma(seed, vocab), family, "cpu")
    bank = W.GemmaSAEBank(device=torch.device("cpu"), family=family, preload=True)
    return inf, bank


def ids(B=2, T=7, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(3, 128, (B, T), generator=g)
    x[:, 0] = 2                                   # BOS
    return x


def raw_logits(m, x):
    with torch.no_grad():
        return m(input_ids=x, use_cache=False).logits


def site_logits(sm, x, transform):
    with torch.no_grad(), sm.patch_sites(transform):
        return sm(x, return_all_logits=True)[0]


# ------------------------------------------------------------------------------------------------------ JumpReLU
def test_jumprelu_encode_matches_reference_formula():
    sae, x = rand_sae(3), torch.randn(4, 5, D)
    acts, idx = sae.encode(x)
    pre = x @ sae.encoder.weight.T + sae.encoder.bias
    ref = pre * (pre > sae.threshold)             # circuit-tracer JumpReLU / sae-lens encode_jumprelu
    assert torch.equal(acts, ref)
    assert idx.shape == acts.shape and int(idx[0, 0, -1]) == DS - 1
    assert torch.equal(sae._get_bias_eff(), sae.encoder.bias)          # seed tap = own pre-activation
    xh = sae.decode(acts)
    assert torch.allclose(xh, acts @ sae.decoder.weight.T + sae.decoder_bias)


# ------------------------------------------------------------------------------------------------------ PLT site
def test_plt_open_gate_is_bitwise_identity():
    m, x = tiny_gemma(), ids()
    sm = W.GemmaPLTSiteModel(m, fam("tc"))
    saes = {l: rand_sae(10 + l) for l in range(NL)}

    def open_gate(l, k, xn):
        c = saes[l].encode(xn)[0]
        return xn + (saes[l].decode(c) - saes[l].decode(c))           # c_hat = c: a new tensor, zero edit

    ref = raw_logits(m, x)
    assert torch.equal(site_logits(sm, x, open_gate), ref)
    assert torch.equal(site_logits(sm, x, lambda l, k, xn: xn), ref)  # same object = no edit
    assert torch.equal(site_logits(sm, x, lambda l, k, xn: None), ref)


def test_plt_read_is_tl_ln2_hook_normalized():
    m, x = tiny_gemma(), ids()
    sm = W.GemmaPLTSiteModel(m, fam("tc"))
    seen, pre_in, ln_out = {}, {}, {}
    hs = []
    for l, layer in enumerate(m.model.layers):
        def rec(mod, inp, out, l=l):
            pre_in[l], ln_out[l] = inp[0].detach(), out.detach()      # returns None: output unchanged
        hs.append(layer.pre_feedforward_layernorm.register_forward_hook(rec))
    with torch.no_grad(), sm.capture_sites(lambda l, acts: seen.__setitem__(l, acts[0])):
        sm(x)
    for h in hs:
        h.remove()
    for l in range(NL):
        layer = m.model.layers[l]
        h = pre_in[l].float()
        tl = h / (h.pow(2).mean(-1, keepdim=True) + layer.pre_feedforward_layernorm.eps).sqrt()   # TL RMSNorm form
        assert torch.allclose(seen[l], tl, atol=1e-6, rtol=1e-5)
        assert torch.allclose(seen[l] * (1 + layer.pre_feedforward_layernorm.weight.float()), ln_out[l].float(),
                              atol=1e-6)


def test_plt_edit_enters_at_mlp_out_and_spares_bos():
    m, x = tiny_gemma(), ids()
    sm = W.GemmaPLTSiteModel(m, fam("tc"))
    delta = torch.randn(1, x.shape[1], D) * 0.3

    def edit(l, k, xn):
        return xn + delta if l == 1 else None

    got = site_logits(sm, x, edit)
    d0 = delta.clone()
    d0[:, :1] = 0                                 # BOS restored
    h = m.model.layers[1].post_feedforward_layernorm.register_forward_hook(lambda mod, i, out: out + d0)
    ref = raw_logits(m, x)
    h.remove()
    assert torch.allclose(got, ref, atol=1e-5)
    assert not torch.allclose(got, raw_logits(m, x), atol=1e-3)
    assert torch.allclose(got[:, 0], raw_logits(m, x)[:, 0], atol=1e-6)   # causal: BOS position untouched


# ------------------------------------------------------------------------------------------------------ resid site
def test_res_native_open_gate_identity_and_bos():
    m, x = tiny_gemma(), ids()
    sm = W.GemmaResSiteModel(m, fam("resid"))
    sae = rand_sae(5)

    def open_gate(l, k, h):
        c = sae.encode(h)[0]
        return h + (sae.decode(c) - sae.decode(c))

    assert torch.equal(site_logits(sm, x, open_gate), raw_logits(m, x))


def _caples_reference(m, x, saes, eps_bar, masked=None):
    """Their build_sae_hook_fn(use_mask, mean_mask, use_mean_error=True), re-stated: codes zeroed at BOS, optional
    mask transform of the codes, decode, torch.where(non-BOS, out, value) + mean_error."""
    hs = []
    for l, sae in saes.items():
        def hook(_m, _i, out, l=l, sae=sae):
            v = out[0] if isinstance(out, tuple) else out
            keep = (x != 2)
            c = sae.encode(v)[0] * keep[..., None]
            if masked is not None:
                c = masked(l, c)
            o = torch.where(keep[..., None], sae.decode(c), v) + eps_bar[l][: v.shape[1]]
            return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        hs.append(m.model.layers[l].register_forward_hook(hook))
    try:
        return raw_logits(m, x)
    finally:
        for h in hs:
            h.remove()


def test_res_caples_substrate_matches_their_splice():
    m, x = tiny_gemma(), ids()
    saes = {0: rand_sae(20), 2: rand_sae(22)}
    g = torch.Generator().manual_seed(7)
    eps_bar = {l: torch.randn(x.shape[1], D, generator=g) * 0.1 for l in saes}
    for l in saes:
        eps_bar[l][0] = 0                         # their mean error is exactly 0 at BOS (stream kept there)
    sm = W.GemmaResSiteModel(m, fam("resid"))
    sm.caples = {l: (saes[l], eps_bar[l]) for l in saes}
    ref_open = _caples_reference(m, x, saes, eps_bar)
    got_open = site_logits(sm, x, lambda l, k, h: None)
    assert torch.allclose(got_open, ref_open, atol=1e-5)
    assert not torch.allclose(got_open, raw_logits(m, x), atol=1e-3)
    # a masked circuit: keep half the latents, the rest at a fill mu (their mean interpolation c' = mu + m (c - mu))
    keep = (torch.arange(DS) % 2 == 0).float()
    mu = {l: torch.rand(x.shape[1], DS) * 0.2 for l in saes}

    def masked(l, c):
        return mu[l] + keep * (c - mu[l])

    ref = _caples_reference(m, x, saes, eps_bar, masked)

    def transform(l, k, h):
        if l not in saes:
            return None
        c = saes[l].encode(h)[0]
        return h + (saes[l].decode(masked(l, c)) - saes[l].decode(c))

    assert torch.allclose(site_logits(sm, x, transform), ref, atol=1e-5)


# ------------------------------------------------------------------------------------------------------ real files
def _cached(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        pytest.skip("dictionary not in the local HF cache: %s" % type(e).__name__)


@pytest.mark.parametrize("which,layer", [("plt", 3), ("res", 0), ("res", 6)])
def test_real_dictionary_load_and_encode(which, layer):
    st = _cached(W.load_plt_state if which == "plt" else W.load_res_state, layer)
    W_enc, b_enc, W_dec, b_dec, thr = st
    assert tuple(W_enc.shape) == (W.D_SAE, W.D_MODEL) and tuple(W_dec.shape) == (W.D_SAE, W.D_MODEL)
    assert thr.shape == (W.D_SAE,) and float(thr.min()) >= 0
    sae = W.jumprelu_from(*st)
    x = torch.randn(3, W.D_MODEL) * (1.0 if which == "plt" else 50.0)
    pre = x @ W_enc.T + b_enc
    assert torch.allclose(sae.encode(x)[0], pre * (pre > thr), atol=1e-4)
    c = pre * (pre > thr)
    assert torch.allclose(sae.decode(c), c @ W_dec + b_dec, rtol=1e-4, atol=1e-2)   # fp32 sums of 16k terms


# ------------------------------------------------------------------------------------------------------ D1 (2026-11-05)
@pytest.mark.parametrize("family", [TINY_PLT, TINY_RES])
def test_seed_threshold_reads_jumprelu_buffer(family):
    """Engine change 1: sae.dense.seed_threshold reads the JumpReLUSAE `threshold` buffer through the bank; the inert
    NullSAE slot has none (ReLU / no dictionary); seed_activation(pre, theta) is the dictionary's own encode."""
    from sae.dense import seed_activation, seed_threshold
    inf, bank = tiny_stack(family)
    kind = W.SPECS[family]["kind"]
    l0 = W.SPECS[family]["layers"][-1]
    sae = bank.saes[kind][l0]
    assert isinstance(sae, W.JumpReLUSAE) and "threshold" in dict(sae.named_buffers())
    x = torch.randn(3, 5, D)
    acts = sae.encode(x)[0]
    for idx in (0, 7, DS - 1):
        thr = seed_threshold(bank, l0, kind, idx)
        assert thr is not None and thr == float(sae.threshold[idx]) and thr > 0
        pre = x @ sae.encoder.weight[idx] + sae.encoder.bias[idx]
        assert torch.allclose(seed_activation(pre, thr), acts[..., idx], atol=1e-6)
        assert bool(((acts[..., idx] != 0) <= (pre > thr)).all())       # fires only above theta, not above 0
    if family == TINY_RES:
        assert isinstance(bank.saes["resid"][1], W.NullSAE) and seed_threshold(bank, 1, "resid", 0) is None
    assert bank.mean_skip_positions == 1


@pytest.mark.parametrize("family", [TINY_PLT, TINY_RES])
def test_pooled_means_skip_bos(family, monkeypatch):
    """Engine change 3: with mean_skip_positions = 1 the pooled means (eval.floors.collect_site_means) are the mean
    over positions >= 1 of the valid slots; 081's Engine.site_means weights its chunks by those counts, so chunked
    means over ragged (right-padded) prompts equal the one-shot mean over every valid non-BOS position."""
    from eval.floors import collect_site_means
    from sae.dense import sparse_topk_to_dense
    inf, bank = tiny_stack(family)
    sites = set(bank.real_sites)
    x = ids(B=5, T=7, seed=4)
    valid = torch.ones(5, 7, dtype=torch.bool)
    valid[1, 5:] = False
    valid[3, 3:] = False
    dense = {}

    def cb(l, acts):
        s = (l, W.SPECS[family]["kind"])
        if s in sites:
            ta, ti = bank.encode(acts[0], s[1], l)
            dense[s] = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=torch.float32)

    with torch.no_grad(), inf.model.capture_sites(cb, capture=False):
        inf.model(x)
    keep = valid.clone()
    keep[:, 0] = False
    got = collect_site_means(inf, bank, x, sites, valid=valid)
    for s in sites:
        ref = dense[s][keep].mean(0)
        with_bos = dense[s][valid].mean(0)
        assert torch.allclose(got[s], ref, atol=1e-6), s
        assert not torch.allclose(got[s], with_bos, atol=1e-4), s       # BOS really moves the mean
    _, E = W._import_081()
    monkeypatch.setattr(E, "MEANS_BS", 2)                               # chunks of 2, 2, 1 prompts
    eng = W.make_engine(family, shared=(inf, bank))
    sm = eng.site_means(x, valid)
    for s in sites:
        assert torch.allclose(sm[s], dense[s][keep].mean(0), atol=1e-6), s


def test_caples_splice_once_under_capture_and_patch():
    """Inference.forward registers the capture hooks and, with a patcher, the patch hooks: the Caples splice must run
    once per layer (the patch hook's), and a capture-only forward or a plain call is the spliced model too."""
    from model.hooks import multi_patch
    inf, bank = tiny_stack(TINY_RES)
    m, x = inf.model.hf, ids()
    saes = {l: bank.saes["resid"][l] for l in RES_LAYERS}
    g = torch.Generator().manual_seed(9)
    eps_bar = {l: torch.randn(x.shape[1], D, generator=g) * 0.1 for l in saes}
    for l in saes:
        eps_bar[l][0] = 0
    W.apply_caples(inf, bank, dict(err=eps_bar))
    ref_open = _caples_reference(m, x, saes, eps_bar)
    keep = (torch.arange(DS) % 3 == 0).float()
    mu = {l: torch.rand(x.shape[1], DS) * 0.2 for l in saes}
    ref_mask = _caples_reference(m, x, saes, eps_bar, lambda l, c: mu[l] + keep * (c - mu[l]))

    def transform(l, k, h):
        if l not in saes:
            return None
        c = saes[l].encode(h)[0]
        return h + (saes[l].decode(mu[l] + keep * (c - mu[l])) - saes[l].decode(c))

    seen = {}
    _, lg, _ = inf.forward(x, tokenize_final=False, return_activations=False, all_logits=True,
                           activations_callback=lambda l, a: seen.__setitem__(l, a[0]))
    assert torch.allclose(lg, ref_open, atol=1e-5)                       # capture only
    with torch.no_grad():
        assert torch.allclose(inf.model(x, return_all_logits=True)[0], ref_open, atol=1e-5)   # plain call
    _, lg2, _ = inf.forward(x, tokenize_final=False, return_activations=False, all_logits=True,
                            patcher=lambda mdl: multi_patch(mdl, transform),
                            activations_callback=lambda l, a: seen.__setitem__(("p", l), a[0]))
    assert torch.allclose(lg2, ref_mask, atol=1e-5)                      # capture + patch: one splice
    assert torch.equal(seen[("p", 0)], seen[0])                          # capture still sees the site input
    W.apply_caples(inf, bank, None)
    _, lg3, _ = inf.forward(x, tokenize_final=False, return_activations=False, all_logits=True)
    assert torch.equal(lg3, raw_logits(m, x))                            # native: the model itself


# ------------------------------------------------------------------------------------------------------ D2 tiny fits
def _fit_tiny(tmp_path, family, mode, means, data, task_kw=None, caples_means=None):
    os.environ.pop("STOP_AT", None)
    os.environ.pop("STOP_EPOCH", None)
    inf, bank = tiny_stack(family, vocab=GEMMA_VOCAB)
    eng = W.make_engine(family, mode, means, caples_means, shared=(inf, bank))
    logs = []
    kw = dict(family=family, data=data, mode=mode, means=means, steps=6, out=tmp_path, eng=eng, log=logs.append,
              caples_means=caples_means, task_kw=task_kw)
    rows = W.run(["rc"], [1e-3], **kw)
    assert len(rows) == 1
    r = rows[0]
    circ = tmp_path / r["circuit"]
    assert circ.exists()
    c = json.loads(circ.read_text())
    assert c["family"] == family and c["wcm"] == W.v3_logit() and c["means"] == means
    assert c["wcm"]["lr_schedule"] == "geometric" and c["wcm"]["rank_weight"] == 0.0
    f = c["fit"]
    assert f.get("sparse_fill_calls", 0) > 0 and f.get("resample_weight") == 1.0 and f.get("resample_draws", 0) > 0
    assert not f.get("rank_weight")
    assert r["task_full"] is not None and r["n_latents"] >= 0
    n_lines = len((tmp_path / "eval.jsonl").read_text().splitlines())
    rows2 = W.run(["rc"], [1e-3], **kw)                                # resumable: the scored key is skipped
    assert rows2 == [] and any(s.startswith("skip (scored)") for s in logs)
    assert len((tmp_path / "eval.jsonl").read_text().splitlines()) == n_lines
    print("RESULT tiny fit %s %s %s %s: %d latents, task_full %s Z %s A %s C %s nec %s, fit %.1fs score %.1fs, "
          "c_fill %s" % (family, data, r["substrate"], means, r["n_latents"], r.get("task_full"), r.get("task_free0"),
                         r.get("task_freeM_topk"), r.get("task_freeN_topk"), r.get("task_nec"), r["t_fit"],
                         r["t_score"], r["c_fill"]))
    return eng, r


CAPLES_TINY = dict(n=160, mean_batches=2, n_held=16)                 # 10 batches; training 6; mean set 32 prompts


def test_tiny_fit_plt_sfc_pooled(tmp_path):
    eng, r = _fit_tiny(tmp_path, TINY_PLT, "native", "pooled", "sfc")
    assert r["n_train"] == 100 and r["n_held"] == 40 and r["substrate"] == "native"


def test_tiny_fit_gsres_native_per_position(tmp_path):
    eng, r = _fit_tiny(tmp_path, TINY_RES, "native", "per_position", "caples", CAPLES_TINY)
    assert r["n_train"] == 96 and r["n_held"] == 16 and r["c_fill"] == "ours"
    assert set(r["latents_per_site"]) <= {"resid_0", "resid_2"}


def _fake_caples_means(path, T_rows, T=7):
    g = torch.Generator().manual_seed(11)
    mt = torch.tensor([r["cf_tokens"] for r in T_rows], dtype=torch.long)
    src = {}
    for l in RES_LAYERS:
        e = torch.randn(T, D, generator=g) * 0.05
        e[0] = 0
        c = torch.rand(T, DS, generator=g) * 0.1
        c[0] = 0
        src[l] = dict(code=c, err=e)
    torch.save(dict(format=W.CAPLES_FORMAT, meta=dict(label="TEST: random"), layers=list(RES_LAYERS),
                    mean_tokens=mt, theirs=src, exact=src), path)
    return path


def test_tiny_fit_gsres_caples_per_position(tmp_path):
    tok = W.tokenizer()
    T = W.build_task("rc", "caples", tok=tok, family=TINY_RES, **CAPLES_TINY)
    p = _fake_caples_means(tmp_path / "means.pt", T["rows"][:T["caples"]["mean_prompts"]])
    eng, r = _fit_tiny(tmp_path / "fit", TINY_RES, "caples", "per_position", "caples", CAPLES_TINY, str(p))
    assert r["substrate"] == "caples_theirs" and r["c_fill"] == "caples_theirs"
    M = eng.means(eng.batch(T))
    cm = W.load_caples_means(p)
    assert torch.equal(M["C"][(2, "resid")].cpu(), cm["code"][2])      # the C fill is the file's
    assert isinstance(eng.inf.model.caples, dict) and sorted(eng.inf.model.caples) == list(RES_LAYERS)
    bad = _fake_caples_means(tmp_path / "bad.pt", T["rows"][1:T["caples"]["mean_prompts"] + 1])
    eng2 = W.make_engine(TINY_RES, "caples", "per_position", str(bad), shared=tiny_stack(TINY_RES, GEMMA_VOCAB))
    with pytest.raises(ValueError, match="other prompts"):
        eng2.batch(T)


def test_caples_mode_refuses_pooled_means(tmp_path):
    """Daniel 2026-11-05: Caples mode is per-position only (their fill); pooled means are for native mode."""
    with pytest.raises(ValueError, match="per-position only"):
        W.make_engine(TINY_RES, "caples", "pooled", str(tmp_path / "unused.pt"), shared=tiny_stack(TINY_RES))
    eng = W.make_engine(TINY_RES, "native", "pooled", shared=tiny_stack(TINY_RES))   # native: pooled allowed
    assert eng.means_mode == "pooled" and eng.cm is None


def test_caples_heldout_deduplicated_their_sets_untouched():
    """The D2 held-out subset (ours) drops every rc_test record whose clean / patch prefix is a training or mean-set
    prefix, then keeps the first 400 in file order; their training batches and mean set are their filter's, unchanged."""
    tok = W.tokenizer()
    T = W.build_task("rc", "caples", tok=tok)
    n_tr = T["split_meta"]["n_train"]
    rows = T["rows"]
    assert n_tr == 1344 and T["split_meta"]["n_held"] == 400 and T["caples"]["train_batches"] == 84
    theirs = W._caples_filter(W.CAPLES_SRC / "data" / "sva" / "rc_train.json", tok, 7, 3000)
    assert [r["prompt"] for r in rows[:n_tr]] == [e["clean_prefix"] for e in theirs[:n_tr]]       # theirs, in order
    assert [r["cf_prompt"] for r in rows[:640]] == [e["patch_prefix"] for e in theirs[:640]]      # their mean set
    seen = {p for r in rows[:n_tr] for p in (r["prompt"], r["cf_prompt"])}
    assert not any(r["prompt"] in seen or r["cf_prompt"] in seen for r in rows[n_tr:])
    assert T["overlap"]["held_prefix_in_train"] == 0
    print("RESULT held-out dedup: dropped %d rc_test records before the first 400" % T["split_meta"]["held_dropped_dup"])


# ------------------------------------------------------------------------------------------------------ D3 test 4
CAPLES_PY = Path.home() / "venvs" / "caples107" / "bin" / "python"


def _ld(logits, cl, pl):
    r = torch.arange(logits.shape[0])
    return float((logits[r, -1, cl] - logits[r, -1, pl]).mean())


def test_caples_mode_open_gate_ld_equals_theirs(tmp_path):
    """README §4.3 test 4 on a tiny random Gemma-2: caples_means.py runs THEIR utils (caples107: TL 2.11 + sae-lens
    5.3.3, the model converted by TL's own from_pretrained) on a fixture; our caples-mode site model (HF stack) with
    their eps_bar reproduces their run_sae_hook_fn(use_mean_error=True) LD per batch. Also: the full-model and the
    error-dropped LDs agree; their 'exact' means equal our own error-dropped per-position means; their 'theirs' means
    equal the running-mean recursion (n = batch index + 1) over our per-prompt codes, i.e. the bug is theirs, kept."""
    import subprocess
    if not CAPLES_PY.exists():
        pytest.skip("caples107 venv not found")
    m = tiny_gemma(3)
    saes = {l: rand_sae(300 + l) for l in RES_LAYERS}
    nb, bs, T, MB = 12, 16, 7, 4
    g = torch.Generator().manual_seed(5)
    clean = torch.randint(3, 128, (nb, bs, T), generator=g)
    clean[..., 0] = 2
    corr = clean.clone()
    corr[..., 2] = torch.randint(3, 128, (nb, bs), generator=g)
    cl, pl = torch.randint(3, 128, (nb, bs), generator=g), torch.randint(3, 128, (nb, bs), generator=g)
    fx = dict(hf_config=dict(TINY_CFG, vocab_size=128), hf_state={k: v.clone() for k, v in m.state_dict().items()},
              saes={l: dict(W_enc=s.encoder.weight.detach().T.contiguous(), W_dec=s.decoder.weight.detach().T.contiguous(),
                            b_enc=s.encoder.bias.detach().clone(), b_dec=s.decoder_bias.detach().clone(),
                            threshold=s.threshold.clone()) for l, s in saes.items()},
              clean=clean, corr=corr, clean_lab=cl, corr_lab=pl, tokenizer=W.HF_ID)
    torch.save(fx, tmp_path / "fixture.pt")
    out = tmp_path / "means.pt"
    env = dict(os.environ, WANDB_MODE="disabled", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([str(CAPLES_PY), "-I", str(HERE / "caples_means.py"), "--fixture", str(tmp_path / "fixture.pt"),
                        "--mean-batches", str(MB), "--batch-size", str(bs), "--example-length", str(T),
                        "--device", "cpu", "--out", str(out)], capture_output=True, text=True, env=env, timeout=1200)
    assert p.returncode == 0, p.stderr[-4000:]
    d = torch.load(out, map_location="cpu", weights_only=False)
    assert d["format"] == W.CAPLES_FORMAT and d["layers"] == list(RES_LAYERS)
    assert torch.equal(d["mean_tokens"], corr[:MB].reshape(-1, T))
    sm = W.GemmaResSiteModel(m, fam("resid"))
    worst = {}

    def check(name, caples):
        sm.caples = caples
        for k, idx in d["meta"]["eval_batches"].items():
            for j, i in enumerate(idx):
                with torch.no_grad():
                    ours = _ld(sm(clean[i], return_all_logits=True)[0], cl[i], pl[i])
                theirs = d["ld"][k][name][j]
                worst[name] = max(worst.get(name, 0.0), abs(ours - theirs))
                assert abs(ours - theirs) <= 1e-4 * (1 + abs(theirs)), (name, k, i, ours, theirs)

    zero = {l: torch.zeros(T, D) for l in RES_LAYERS}
    check("model", None)
    check("sae", {l: (saes[l], zero[l]) for l in RES_LAYERS})                               # error dropped
    check("sae_mean_error", {l: (saes[l], d["theirs"][l]["err"]) for l in RES_LAYERS})       # test 4
    # their means vs ours, in the error-dropped splice (the substrate their mean functions run in)
    sm.caples = {l: (saes[l], zero[l]) for l in RES_LAYERS}
    flat = corr[:MB].reshape(-1, T)
    codes, errs = {l: [] for l in RES_LAYERS}, {l: [] for l in RES_LAYERS}

    def cb(l, acts):
        if l in saes:
            c = saes[l].encode(acts[0])[0]
            c[:, 0] = 0                                                   # their codes are zeroed at BOS
            e = acts[0] - saes[l].decode(c)
            e[:, 0] = 0                                                   # their error is 0 at BOS (stream kept)
            codes[l].append(c)
            errs[l].append(e)

    with torch.no_grad(), sm.capture_sites(cb, capture=False):
        for i in range(flat.shape[0]):
            sm(flat[i:i + 1])
    for l in RES_LAYERS:
        C_ = torch.cat(codes[l]).double()
        E_ = torch.cat(errs[l]).double()
        assert torch.allclose(d["exact"][l]["code"].double(), C_.mean(0), atol=1e-4), l
        assert torch.allclose(d["exact"][l]["err"].double(), E_.mean(0), atol=1e-4), l
        rc, re_ = torch.zeros(DS, dtype=torch.float64), torch.zeros(D, dtype=torch.float64)
        for i in range(MB):                                               # their running_mean_tensor, n = i + 1
            for j in range(bs):
                rc = rc + (C_[i * bs + j] - rc) / (i + 1)
                re_ = re_ + (E_[i * bs + j] - re_) / (i + 1)
        assert torch.allclose(d["theirs"][l]["code"].double(), rc, atol=1e-4), l
        assert torch.allclose(d["theirs"][l]["err"].double(), re_, atol=1e-4), l
        assert float((d["theirs"][l]["code"] - d["exact"][l]["code"]).abs().max()) > 1e-3   # the bug is visible
    cm = W.load_caples_means(out, "theirs")
    assert sorted(cm["err"]) == list(RES_LAYERS) and cm["mean_tokens"].shape == (MB * bs, T)
    print("RESULT test 4 (tiny): max |LD ours - theirs| %s; their mean LDs %s" % (
        {k: "%.2e" % v for k, v in worst.items()},
        {k: {n: round(sum(v) / len(v), 4) for n, v in dd.items()} for k, dd in d["ld"].items()}))


@pytest.mark.gpu
@pytest.mark.skipif(not os.environ.get("ARENA_D_CAPLES_PT") or not torch.cuda.is_available(),
                    reason="GPU night / pod: set ARENA_D_CAPLES_PT to caples_means.py's real-model file")
def test_real_caples_mode_ld_gpu():
    """[GPU, Thursday night / pod] test 4 on the real Gemma-2-2B: our caples-mode open-gate LD (HF stack, the file's
    eps_bar) vs their run_sae_hook_fn(use_mean_error=True) LD (TL stack) per eval batch. The stacks differ
    numerically (README §4.3 test 3: within 1e-2), so the bound is 1e-2 in fp32 and a labelled 0.15 in bf16."""
    pt = os.environ["ARENA_D_CAPLES_PT"]
    d = torch.load(pt, map_location="cpu", weights_only=False)
    dtype = d["meta"]["args"]["dtype"]
    inf, bank = W.load_model(W.FAM_RES, "cuda", None if dtype == "float32" else dtype)
    cm = W.load_caples_means(pt, "theirs")
    W.apply_caples(inf, bank, cm)
    tok = W.tokenizer()
    a = d["meta"]["args"]
    task = Path(a["task"]).name.replace("_train", "")
    recs = W._caples_filter(W.CAPLES_SRC / "data" / ("%s.json" % a["task"]), tok, a["example_length"], a["N"])
    bs = a["batch_size"]
    tol = 1e-2 if dtype == "float32" else 0.15
    worst = 0.0
    for k in ("first10", "eval"):
        for j, i in enumerate(d["meta"]["eval_batches"][k]):
            b = recs[i * bs:(i + 1) * bs]
            x = torch.tensor([tok(e["clean_prefix"]).input_ids for e in b], device="cuda")
            cl = torch.tensor([W._answer_id(tok, e["clean_answer"]) for e in b], device="cuda")
            pl = torch.tensor([W._answer_id(tok, e["patch_answer"]) for e in b], device="cuda")
            with torch.no_grad():
                ours = _ld(inf.model(x, return_all_logits=True)[0].float(), cl, pl)
            worst = max(worst, abs(ours - d["ld"][k]["sae_mean_error"][j]))
    print("RESULT test 4 (real %s, %s): max |LD ours - theirs| = %.4f" % (task, dtype, worst))
    assert worst <= tol


# ------------------------------------------------------------------------------------------------------ D6 (GPU)
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU night / pod: the real Gemma-2-2B on CUDA")
@pytest.mark.parametrize("family", [W.FAM_PLT, W.FAM_RES])
def test_real_open_gate_identity_gpu(family):
    """[GPU, plan D6] README §4.3 test 1 on the real model: the unsloth bf16 Gemma-2-2B (GEMMA_HF_ID; pod: the google
    weights) through our site model with every real dictionary loaded (bank dtype bf16; PLT: 26 transcoders, gsres:
    5 Gemma Scope SAEs + 21 NullSAE slots, native mode) gives logits BITWISE equal to the raw HF model on 8 SVA rc
    prompts (Caples' filter, length 7): an open-gate transform (c_hat = c: x + (decode(c) - decode(c)), a new tensor
    and a zero edit at every site), an identity transform (returns x itself) and a plain call. The open gate also
    runs every encode / decode on the real dictionaries."""
    import gc
    dtype = os.environ.get("ARENA_D_ID_DTYPE", "bfloat16")
    inf, bank = W.load_model(family, "cuda", dtype)
    kind = W.SPECS[family]["kind"]
    tok = W.tokenizer()
    recs = W._caples_filter(W.CAPLES_SRC / "data" / "sva" / "rc_train.json", tok, 7, 8)
    x = torch.tensor([tok(e["clean_prefix"]).input_ids for e in recs], device="cuda")
    assert x.shape == (8, 7) and bool((x[:, 0] == tok.bos_token_id).all())
    sm, hf = inf.model, inf.model.hf
    n_sites = [0]

    def open_gate(l, k, h):
        assert k == kind
        sae = bank.saes[kind][l]
        c = sae.encode(h)[0]
        n_sites[0] += 1
        return h + (sae.decode(c) - sae.decode(c))

    with torch.no_grad():
        ref = hf(input_ids=x, use_cache=False).logits
        with sm.patch_sites(open_gate):
            got_open = sm(x, return_all_logits=True)[0]
        with sm.patch_sites(lambda l, k, h: h):
            got_id = sm(x, return_all_logits=True)[0]
        got_plain = sm(x, return_all_logits=True)[0]
    assert n_sites[0] == W.N_LAYER, n_sites[0]                         # one site per layer, all visited
    res = {}
    for name, got in (("open_gate", got_open), ("identity", got_id), ("plain", got_plain)):
        res[name] = float((got.float() - ref.float()).abs().max())
    print("RESULT D6 identity (%s, %s, %s): max |dlogit| %s" % (family, W.HF_ID, dtype, res))
    assert all(torch.equal(g, ref) for g in (got_open, got_id, got_plain)), res
    del inf, bank, sm, hf, ref, got_open, got_id, got_plain
    gc.collect()
    torch.cuda.empty_cache()
