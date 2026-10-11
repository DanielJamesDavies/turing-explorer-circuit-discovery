"""DAN-34 / DAN-35 / DAN-74: the public-model adapters (Pythia-70m, GPT-2 small).

Every check runs on CPU against the cached HF checkpoints and SAEs; a family
whose files are not in the local HF cache is skipped (never downloaded). The
references are independent of the adapter: plain PyTorch hooks on the HF
modules at the SAEs' training hook points, and 053's layer-norm arithmetic
re-typed here.

  * the unedited engine forward equals the HF model's logits, bitwise;
  * a zero decoded delta (the engine's own edit path) is the identity, bitwise,
    at every site;
  * the walker hands each transform exactly the SAE input at the hook point,
    so the engine's seed tap equals x @ w + b on the reference stream, bitwise;
  * an edit at every site shifts the downstream residual stream by the decoded
    delta (x std for GPT-2), and only where the architecture says it should:
    Pythia's parallel residual leaves the same-layer mlp untouched by an attn
    edit; GPT-2's resid_mid edit reaches the residual stream, not only the MLP
    input (the DAN-68 bug);
  * Top-K: k slots per token at every site (non-zeros: exactly k on Pythia,
    <= k on GPT-2 where ReLU zeros a few, as in 053's audit);
  * bank.preact: relu + top-k reproduces bank.encode exactly at every site.

GPT-2 layer 10 is not in the local cache for any of the four SAE repos; its
sites are skipped here and listed by the audit script.
"""

import os

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from model.families import FAMILIES, GPT2_SMALL, PYTHIA_70M, family_upstream_sites  # noqa: E402
from model.hooks import capture_activations, multi_patch, patch  # noqa: E402

B, T = 2, 24
DELTA_LATENT_SCALE = 5.0


def _cached(repo: str, filename: str) -> bool:
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(repo, filename)
    return isinstance(p, str) and os.path.exists(p)


def _site_cached(family, kind, layer) -> bool:
    repo, weights, _ = family.sae.files(kind, layer)
    return _cached(repo, weights)


def _family_available(name: str) -> bool:
    fam = FAMILIES[name]
    return _cached(fam.hf_model_id, "config.json") and _site_cached(fam, fam.kinds[0], 0)


FAMILY_PARAMS = [
    pytest.param(name, marks=pytest.mark.skipif(not _family_available(name),
                                                 reason=f"{name} not in the HF cache"))
    for name in (PYTHIA_70M, GPT2_SMALL)
]


class _Env:
    def __init__(self, name):
        from model.inference import Inference
        from sae.bank import SAEBank
        torch.manual_seed(0)
        self.family = FAMILIES[name]
        self.inference = Inference(torch.device("cpu"), family=name)
        self.bank = SAEBank(device=torch.device("cpu"), family=name, preload=False)
        self.hf = self.inference.model.hf
        self.tokens = _text_tokens(name)
        self.sites = [(l, k) for l in range(self.family.n_layer) for k in self.family.kinds
                      if _site_cached(self.family, k, l)]
        self.ref = _reference_streams(self, self.tokens)

    def norm(self, x):
        """053's gpt2saes.norm_in, re-typed: (x_n, std + eps)."""
        if self.family.sae.normalize != "layer_norm":
            return x, None
        mu = x.mean(dim=-1, keepdim=True)
        xc = x - mu
        std = xc.std(dim=-1, keepdim=True)
        eps = self.family.sae.norm_eps
        return xc / (std + eps), std + eps


def _text_tokens(name):
    """B windows of T tokens from cached wikitext (random ids if unavailable)."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(FAMILIES[name].tokenizer_id)
        path = hf_hub_download("Salesforce/wikitext", repo_type="dataset",
                               filename="wikitext-103-raw-v1/test-00000-of-00001.parquet")
        texts = [t for t in pq.read_table(path, columns=["text"])["text"].to_pylist()[:400]
                 if len(t) > 200]
        ids = tok(" ".join(texts), add_special_tokens=False)["input_ids"]
        return torch.tensor(ids[:B * T]).view(B, T)
    except Exception:  # noqa: BLE001  (no dataset in the cache: still a valid forward)
        return torch.randint(0, 50000, (B, T), generator=torch.Generator().manual_seed(0))


def _hook_points(env):
    """(module, mode) per site: where the SAE reads, from the HF modules."""
    pts = {}
    if env.family.name == PYTHIA_70M:
        for l, layer in enumerate(env.hf.gpt_neox.layers):
            pts[(l, "attn")] = (layer.attention, "out0")
            pts[(l, "mlp")] = (layer.mlp, "out")
            pts[(l, "resid")] = (layer, "out0")
    else:
        for l, block in enumerate(env.hf.transformer.h):
            pts[(l, "attn")] = (block.attn, "out0")
            pts[(l, "resid_mid")] = (block.ln_2, "in0")      # the block's own attn_output + residual
            pts[(l, "mlp")] = (block.mlp, "out")
            pts[(l, "resid")] = (block, "out0")
    return pts


def _next_consumer(env, layer):
    """The module whose INPUT is resid(layer): next layer, or the final norm."""
    if env.family.name == PYTHIA_70M:
        layers = env.hf.gpt_neox.layers
        return layers[layer + 1] if layer + 1 < len(layers) else env.hf.gpt_neox.final_layer_norm
    blocks = env.hf.transformer.h
    return blocks[layer + 1] if layer + 1 < len(blocks) else env.hf.transformer.ln_f


def _record(module, mode, store, key):
    if mode == "in0":
        return module.register_forward_pre_hook(
            lambda m, a, kw: store.__setitem__(key, (a[0] if a else kw["hidden_states"]).detach().clone()),
            with_kwargs=True)

    def hook(m, i, o):
        x = o[0] if (mode == "out0" and isinstance(o, tuple)) else o
        store[key] = x.detach().clone()
    return module.register_forward_hook(hook)


def _reference_streams(env, tokens, transform=None):
    """Raw HF streams at every hook point (+ next-consumer inputs and, for
    GPT-2, ln_2 inputs), optionally with an engine transform active. The
    recording hooks are registered AFTER the walker's, so they see edits."""
    store = {}
    ctx = multi_patch(env.inference.model, transform) if transform else None
    handles = []
    if ctx is not None:
        ctx.__enter__()
    try:
        for site, (mod, mode) in _hook_points(env).items():
            handles.append(_record(mod, mode, store, site))
        for l in range(env.family.n_layer):
            handles.append(_record(_next_consumer(env, l), "in0", store, ("next", l)))
        with torch.no_grad():
            store["logits"] = env.hf(input_ids=tokens, use_cache=False).logits.detach().clone()
    finally:
        for h in handles:
            h.remove()
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return store


@pytest.fixture(scope="module", params=FAMILY_PARAMS)
def env(request):
    return _Env(request.param)


# --------------------------------------------------------------------------- forward identity

def test_unedited_engine_forward_matches_hf_logits_bitwise(env):
    tok = env.tokens
    _, lg_all, _ = env.inference.forward(tok, all_logits=True, return_activations=False,
                                         tokenize_final=False)
    assert torch.equal(lg_all, env.ref["logits"])
    # last-position path == HF's own logits_to_keep=1 path
    _, lg_last, _ = env.inference.forward(tok, return_activations=False, tokenize_final=False)
    with torch.no_grad():
        ref_last = env.hf(input_ids=tok, use_cache=False, logits_to_keep=1).logits
    assert torch.equal(lg_last, ref_last)
    # walker + capture active, transform returns its input: still bitwise
    out = env.inference.forward(tok, all_logits=True, return_activations=True,
                                tokenize_final=False,
                                patcher=lambda m: multi_patch(m, lambda l, k, x: x))
    assert torch.equal(out[1], env.ref["logits"])


def test_engine_mask_patcher_natural_pass_is_untouched(env):
    """The engine's own natural pass (LearnedMaskPatcher, no thetas)."""
    from circuit.instrument.learned_mask import LearnedMaskPatcher
    l, k = env.sites[len(env.sites) // 2]
    w, b = env.bank.preact_vector(k, l, 0)
    p = LearnedMaskPatcher(env.bank, {}, l, k, w, b)
    _, lg, _ = env.inference.forward(env.tokens, patcher=p, all_logits=True,
                                     return_activations=False, tokenize_final=False)
    assert torch.equal(lg, env.ref["logits"])


def test_zero_decoded_delta_is_identity_at_every_site(env):
    zero = torch.zeros(env.bank.d_sae)

    def transform(l, k, x):
        if (l, k) not in env.sites:
            return None
        return x + env.bank.decode(zero.expand(*x.shape[:-1], -1), k, l, add_bias=False)

    out = _reference_streams(env, env.tokens, transform)
    for site in env.sites:
        assert torch.equal(out[site], env.ref[site]), site
    assert torch.equal(out["logits"], env.ref["logits"])


# --------------------------------------------------------------------------- site input / tap

def test_capture_gives_sae_input_at_every_hook_point(env):
    with torch.no_grad(), capture_activations(env.inference.model) as acts:
        env.inference.model(env.tokens, return_all_logits=True)
    kinds = env.family.kinds
    assert acts.tensor.shape == (B, env.family.n_layer, len(kinds), T, env.family.d_model)
    for (l, k) in env.sites:
        x_ref, _ = env.norm(env.ref[(l, k)])
        assert torch.equal(acts.tensor[:, l, kinds.index(k)], x_ref), (l, k)


def test_seed_tap_equals_x_w_plus_b_bitwise_on_reference_stream(env):
    from circuit.instrument.learned_mask import _natural
    for (l, k) in env.sites:
        x_ref, _ = env.norm(env.ref[(l, k)])
        pre = env.bank.preact(x_ref, k, l)
        idx = int(pre.reshape(-1, pre.shape[-1]).max(0).values.argmax())
        w, b = env.bank.preact_vector(k, l, idx)
        tap = _natural(env.inference, env.bank, env.tokens, l, k, w, b)
        assert torch.equal(tap, x_ref @ w + b), (l, k)
        # and it IS the dictionary's pre-activation: (x - b_dec) W_enc^T + b_enc
        sae = env.bank.saes[k][l]
        direct = (x_ref - sae.decoder_bias) @ sae.encoder.weight[idx] + sae.encoder.bias[idx]
        torch.testing.assert_close(tap, direct, rtol=1e-5, atol=1e-4)
        env.bank.release(k, l)


# --------------------------------------------------------------------------- edits enter the stream

def _downstream_checks(env, l, k):
    """[(stream key or callable, expects shift?)] after an edit at (l, k)."""
    nxt = ("next", l)
    if env.family.name == PYTHIA_70M:
        if k in ("attn", "mlp"):
            other = "mlp" if k == "attn" else "attn"
            # parallel residual: the same-layer sibling does not see the edit
            return [((l, "resid"), True), ((l, other), False), (nxt, True)]
        return [(nxt, True)]
    if k == "attn":
        return [((l, "resid_mid"), True), (nxt, None)]
    if k == "resid_mid":
        # residual stream = block output - mlp output; the 053 bug left it clean
        return [((l, "resid_mid"), True), ("resid_minus_mlp", True)]
    if k == "mlp":
        return [((l, "resid_mid"), False), ((l, "resid"), True), (nxt, True)]
    return [(nxt, True)]


def test_edit_shifts_downstream_residual_by_decoded_delta(env):
    for (l, k) in env.sites:
        sae_dec = env.bank.saes[k][l]
        latent = int(sae_dec.decoder.weight.norm(dim=0).argmax())
        code = torch.zeros(env.bank.d_sae)
        code[latent] = DELTA_LATENT_SCALE
        d_site = env.bank.decode(code, k, l, add_bias=False)            # [d_model], site space
        _, scale = env.norm(env.ref[(l, k)])
        shift = d_site if scale is None else d_site * scale              # model space, per token

        def transform(ll, kk, x, _site=(l, k)):
            return x + d_site if (ll, kk) == _site else None

        out = _reference_streams(env, env.tokens, transform)
        for key, expect in _downstream_checks(env, l, k):
            if key == "resid_minus_mlp":
                got = (out[(l, "resid")] - out[(l, "mlp")]) - (env.ref[(l, "resid")] - env.ref[(l, "mlp")])
                base = env.ref[(l, "resid")]
            else:
                got = out[key] - env.ref[key]
                base = env.ref[key]
            if expect is None:
                continue
            # fp32 rounding of a few adds on this token's stream
            tol = 1e-5 * base.abs().amax(-1, keepdim=True) + 1e-5
            if expect:
                big = shift.expand_as(got).abs().amax(-1, keepdim=True) > 100 * tol
                assert float(big.float().mean()) > 0.9, (l, k, "delta too small to test")
                err = (got - shift.expand_as(got)).abs()
                assert bool((err <= tol).all()), (
                    f"{(l, k)} -> {key}: shift != decoded delta "
                    f"(max err {float(err.max()):.3g}, max shift {float(shift.abs().max()):.3g})")
            else:
                assert torch.equal(out[key], env.ref[key]), f"{(l, k)} leaked into {key}"
        env.bank.release(k, l)


def test_adapter_patch_replaces_one_site(env):
    """patch() writes a site-space value; a later reader at the same site sees
    it (exactly on Pythia; to rounding through GPT-2's layer-norm round trip,
    so the value must itself be a valid normalised activation: mean 0, std 1)."""
    l, k = env.sites[0]
    x_site, _ = env.norm(env.ref[(l, k)])
    value = x_site.flip(1)                     # a real activation, other positions
    seen = {}
    with torch.no_grad(), patch(env.inference.model, l, k, value):
        with multi_patch(env.inference.model,
                         lambda ll, kk, x: seen.__setitem__((ll, kk), x.clone()) or None):
            env.inference.model(env.tokens, return_all_logits=True)
    if env.family.sae.normalize is None:
        assert torch.equal(seen[(l, k)], value)
    else:
        torch.testing.assert_close(seen[(l, k)], value, rtol=0, atol=1e-4)


# --------------------------------------------------------------------------- Top-K contract

def test_topk_slots_equal_k_at_every_site(env):
    k_sae = env.bank.k
    for (l, k) in env.sites:
        x, _ = env.norm(env.ref[(l, k)])
        ta, ti = env.bank.encode(x, k, l)
        assert ta.shape[-1] == k_sae and ti.shape[-1] == k_sae
        nz = (ta > 0).sum(-1)
        assert int(nz.max()) <= k_sae
        if env.family.name == PYTHIA_70M:
            assert bool((nz == k_sae).all()), (l, k, float(nz.float().mean()))
        else:
            assert float(nz.float().mean()) >= 0.9 * k_sae, (l, k, float(nz.float().mean()))
        env.bank.release(k, l)


@pytest.mark.parametrize("grad", [False, True])
def test_preact_topk_reproduces_encode_at_every_site(env, grad):
    for (l, k) in env.sites:
        x, _ = env.norm(env.ref[(l, k)])
        with torch.set_grad_enabled(grad):
            ta, ti = env.bank.encode(x, k, l)
            v, i = torch.relu(env.bank.preact(x, k, l)).topk(env.bank.k, dim=-1, sorted=False)
        # dense codes: identical indices AND values for every non-zero slot
        # (zero-valued slots may tie differently and carry no code)
        assert torch.equal(_dense(ta, ti, env.bank.d_sae), _dense(v, i, env.bank.d_sae)), (l, k)
        if env.family.name == PYTHIA_70M:          # all k slots positive: indices equal outright
            assert torch.equal(ti.sort(-1).values, i.sort(-1).values), (l, k)
        env.bank.release(k, l)


def _dense(vals, idx, d_sae):
    from sae.dense import sparse_topk_to_dense
    return sparse_topk_to_dense(vals.detach(), idx, d_sae)


# --------------------------------------------------------------------------- structure (no weights)

def test_upstream_sites_follow_each_architecture():
    from eval.ablation_faithfulness import upstream_sites

    class _Bank:
        def __init__(self, fam):
            self.kinds = list(fam.kinds)
            self.same_layer_upstream = dict(fam.same_layer_upstream)

    py, g2 = _Bank(FAMILIES[PYTHIA_70M]), _Bank(FAMILIES[GPT2_SMALL])
    # Pythia: parallel residual — mlp(L) does not read attn(L); resid reads both
    assert upstream_sites(py, 2, "mlp") == {(l, kk) for l in range(2) for kk in py.kinds}
    assert upstream_sites(py, 2, "attn") == upstream_sites(py, 2, "mlp")
    assert upstream_sites(py, 2, "resid") - upstream_sites(py, 2, "mlp") == {(2, "attn"), (2, "mlp")}
    assert upstream_sites(py, 0, "mlp") == set()
    # GPT-2: attn < resid_mid < mlp < resid
    same = lambda kind: {kk for (l, kk) in upstream_sites(g2, 3, kind) if l == 3}  # noqa: E731
    assert same("attn") == set() and same("resid_mid") == {"attn"}
    assert same("mlp") == {"attn", "resid_mid"} and same("resid") == {"attn", "resid_mid", "mlp"}
    assert upstream_sites(g2, 3, "resid") == family_upstream_sites(FAMILIES[GPT2_SMALL], 3, "resid")


def test_turingllm_upstream_rule_unchanged(mock_sae_bank):
    from eval.ablation_faithfulness import upstream_sites
    assert upstream_sites(mock_sae_bank, 1, "mlp") == {(0, "attn"), (0, "mlp"), (0, "resid"), (1, "attn")}
    fam = FAMILIES["turingllm"]
    for l in range(3):
        for k in fam.kinds:
            class _B:
                kinds = list(fam.kinds)
            assert upstream_sites(_B(), l, k) == family_upstream_sites(fam, l, k)


def test_default_family_builds_turingllm_classes():
    from model.families import resolve_family
    from model.inference import Inference
    from sae.bank import SAEBank
    assert resolve_family() == "turingllm"
    assert type(Inference.__new__(Inference)) is Inference
    assert type(SAEBank.__new__(SAEBank)) is SAEBank


def test_turingllm_bank_preact_api_matches_encode_and_tap():
    """The base SAEBank's preact / preact_vector on a small random SAE:
    preact_vector is bitwise the engine's (encoder row, b_eff), and
    relu(preact).topk reproduces encode."""
    from sae.bank import SAEBank
    from sae.topk_sae import SAE
    torch.manual_seed(1)
    bank = object.__new__(SAEBank)
    bank.kinds, bank.n_layer, bank.d_sae, bank.k = ["attn", "mlp", "resid"], 1, 64, 8
    bank.layer_device_map = {0: torch.device("cpu")}
    sae = SAE(16, 64, 8, device=torch.device("cpu"))
    with torch.no_grad():
        sae.decoder_bias.normal_()
    bank.saes = {kk: [sae] for kk in bank.kinds}
    x = torch.randn(3, 5, 16)
    w, b = bank.preact_vector("mlp", 0, 7)
    assert torch.equal(w, sae.encoder.weight[7]) and torch.equal(b, sae._get_bias_eff()[7])
    for grad in (False, True):
        with torch.set_grad_enabled(grad):
            ta, ti = bank.encode(x, "mlp", 0)
            v, i = torch.relu(bank.preact(x, "mlp", 0)).topk(8, dim=-1, sorted=False)
        assert torch.equal(_dense(ta, ti, 64), _dense(v, i, 64))
    torch.testing.assert_close(bank.preact(x, "mlp", 0)[..., 7], x @ w + b)
