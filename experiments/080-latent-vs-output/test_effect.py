"""CPU checks of next_token_effect.py's ablation and bookkeeping on a tiny synthetic model (no GPU, no engine):

  PYTHONPATH=src python -m pytest experiments/080-latent-vs-output/test_effect.py -q

The model is tests/conftest.py's MockModel (2 blocks, attn / mlp / resid sites hooked by model.hooks.multi_patch, as
the real engine is), with a token embedding and an unembedding in front and behind; the SAE bank has a ReLU Top-K
encoder per site and a decoder whose decode() takes add_bias like SAEBank's.
"""
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import next_token_effect as NE  # noqa: E402
from tests.conftest import D_MODEL, MockModel  # noqa: E402

D_SAE, K, V, T = 24, 4, 40, 8
SITE = (1, "attn")


class Bank:
    kinds = ["attn", "mlp", "resid"]
    d_sae, k, n_layer = D_SAE, K, 2
    device = torch.device("cpu")

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W_enc = {(l, kd): torch.randn(D_SAE, D_MODEL, generator=g) for l in range(2) for kd in self.kinds}
        self.b_enc = {s: torch.zeros(D_SAE) for s in self.W_enc}
        self.W_dec = {s: torch.randn(D_MODEL, D_SAE, generator=g) / D_MODEL ** 0.5 for s in self.W_enc}
        self.b_dec = {s: 0.3 * torch.randn(D_MODEL, generator=g) for s in self.W_enc}

    def encode(self, x, kind, layer):
        pre = torch.relu(x.float() @ self.W_enc[(layer, kind)].T + self.b_enc[(layer, kind)])
        return pre.topk(K, dim=-1)

    def decode(self, latents, kind, layer, add_bias=True):
        out = latents.float() @ self.W_dec[(layer, kind)].T
        return out + self.b_dec[(layer, kind)] if add_bias else out


class Inference:
    def __init__(self, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.model = MockModel()
        self.emb = torch.randn(V, D_MODEL, generator=g)
        self.W_U = torch.randn(D_MODEL, V, generator=g)

    def disable_compile(self):
        pass

    def enable_compile(self):
        pass

    def forward(self, tokens, patcher=None, **kw):
        x = self.emb[tokens.long()]
        with (patcher(self.model) if patcher else nullcontext()):
            y = self.model(x)
        return tokens, y @ self.W_U, None


def _active_and_silent(bank, inf, tokens):
    """A latent active somewhere at SITE on these tokens, and one silent everywhere (its encoder bias forced very
    negative)."""
    acts = {}

    class Grab:
        def __call__(self, model):
            from model.hooks import multi_patch
            return multi_patch(model, self.t)

        def t(self, l, kd, x):
            if (l, kd) == SITE:
                ta, ti = bank.encode(x, kd, l)
                acts["ti"] = ti
            return x
    inf.forward(tokens, patcher=Grab())
    counts = torch.bincount(acts["ti"].reshape(-1), minlength=D_SAE)
    active = int(counts.argmax())
    silent = int((counts == 0).nonzero()[0]) if (counts == 0).any() else (active + 1) % D_SAE
    bank.b_enc[SITE][silent] = -1e6
    return active, silent


def _setup():
    bank, inf = Bank(), Inference()
    g = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, V, (6, T), generator=g)
    active, silent = _active_and_silent(bank, inf, tokens)
    return bank, inf, tokens, active, silent


def test_silent_latent_changes_nothing_exactly():
    bank, inf, tokens, active, silent = _setup()
    p = NE.TargetAblation(bank, [None, (SITE[0], SITE[1], silent)])
    x = torch.randn(2, T, D_MODEL)
    assert p.transform(SITE[0], SITE[1], x) is x                # untouched: the very same tensor
    nxt, anc = torch.randint(0, V, (6,)), torch.full((6,), T - 1)
    res = NE.run_pack(inf, bank, [("1.attn.%d" % silent, (SITE[0], SITE[1], silent), tokens, nxt, anc)])
    r = res["1.attn.%d" % silent]
    assert r["delta"] == [0.0] * 6 and r["kl"] == [0.0] * 6 and r["act"] == [0.0] * 6


def test_edit_is_only_the_decoded_difference():
    """Ablated row: x - a_f(x, t) d_f at every position (no decoder bias, the SAE error kept); clean rows and other
    sites untouched, bit for bit."""
    bank, inf, tokens, active, silent = _setup()
    cap = {}

    class Grab:                                                  # the model's own stream at SITE
        def __call__(self, model):
            from model.hooks import multi_patch
            return multi_patch(model, self.t)

        def t(self, l, kd, x_):
            if (l, kd) == SITE:
                cap["x"] = x_.clone()
            return x_
    inf.forward(tokens, patcher=Grab())
    order = torch.argsort(-(bank.encode(cap["x"], SITE[1], SITE[0])[1] == active).sum((1, 2)))
    x = cap["x"][order[:3]]                                      # the rows where the latent fires most first
    x = x[[1, 0, 2]]                                             # ... with the most active one as the ablated row
    p = NE.TargetAblation(bank, [None, (SITE[0], SITE[1], active), None])
    out = p.transform(SITE[0], SITE[1], x)
    from sae.dense import target_latent_activations
    ta, ti = bank.encode(x[1:2], SITE[1], SITE[0])
    a = target_latent_activations(ta, ti, active)[0]             # [T], the production read
    assert bool((a > 0).any())
    expect = x[1] - a[:, None] * bank.W_dec[SITE][:, active][None, :]
    assert torch.allclose(out[1], expect, atol=1e-5)
    assert torch.equal(out[0], x[0]) and torch.equal(out[2], x[2])
    zero_t = (a == 0)
    assert torch.equal(out[1][zero_t], x[1][zero_t])             # positions where the latent is off: unchanged
    assert torch.allclose(p.acts[1], a)
    y = torch.randn(3, T, D_MODEL)
    assert p.transform(0, "attn", y) is y and p.transform(1, "mlp", y) is y


def test_run_pack_matches_two_separate_forwards_and_signs():
    """delta = log p_clean - log p_ablated of the actual next token, from the interleaved batch, equals the same two
    passes run separately; summarise() turns it into effect_abs / effect_signed / kl."""
    bank, inf, tokens, active, silent = _setup()
    nxt = torch.randint(0, V, (6,), generator=torch.Generator().manual_seed(9))
    anc = torch.tensor([7, 6, 5, 7, 3, 7])
    key = "1.attn.%d" % active
    r = NE.run_pack(inf, bank, [(key, (SITE[0], SITE[1], active), tokens, nxt, anc)])[key]
    clean = torch.log_softmax(inf.forward(tokens)[1], -1)
    abl = torch.log_softmax(inf.forward(tokens, patcher=NE.TargetAblation(bank, [(SITE[0], SITE[1], active)] * 6))[1],
                            -1)
    rr = torch.arange(6)
    d = (clean[rr, anc, nxt] - abl[rr, anc, nxt]).tolist()
    assert r["delta"] == pytest.approx(d, abs=1e-5)
    kl = (clean[rr, anc].exp() * (clean[rr, anc] - abl[rr, anc])).sum(-1).tolist()
    assert r["kl"] == pytest.approx(kl, abs=1e-5) and all(v >= -1e-6 for v in r["kl"])
    assert any(abs(v) > 1e-4 for v in r["delta"])
    row = NE.summarise(key, r, dict(n_excluded=0, n_clamped=0))
    assert row["effect_abs"] == pytest.approx(sum(abs(v) for v in d) / 6, abs=1e-5)
    assert row["effect_signed"] == pytest.approx(sum(d) / 6, abs=1e-5)
    assert row["n_contexts"] == 6 and row["band"] == "L0-4" and row["kind"] == "attn"


def test_packing_targets_together_changes_nothing():
    bank, inf, tokens, active, silent = _setup()
    nxt, anc = torch.randint(0, V, (6,)), torch.full((6,), 5)
    a = ("1.attn.%d" % active, (SITE[0], SITE[1], active), tokens[:3], nxt[:3], anc[:3])
    b = ("0.mlp.2", (0, "mlp", 2), tokens[3:], nxt[3:], anc[3:])
    both = NE.run_pack(inf, bank, [a, b])
    for job in (a, b):
        alone = NE.run_pack(inf, bank, [job])[job[0]]
        assert both[job[0]]["delta"] == pytest.approx(alone["delta"], abs=1e-6)


def _rec():
    g = torch.Generator().manual_seed(0)
    pos = torch.randint(1, V, (10, 64), generator=g)
    tgt = torch.cat([pos[:, 1:], torch.randint(1, V, (10, 1), generator=g)], 1)
    tgt[4, 30:] = 0                                             # a short sequence: zero padding from position 30
    arg = torch.tensor([3, 63, 10, 70, 31, 5, 6, 7, 8, 9])      # 70: out of range, clamped to 63
    peak = torch.linspace(10, 1, 10)
    return dict(strong=dict(pos=pos, tgt=tgt, arg=arg, peak=peak, train=list(range(10))), D_train=[0, 1, 2, 3, 4, 5])


def test_contexts_of_uses_the_training_contexts_and_handles_edges():
    rec = _rec()
    tok, nxt, anc, info = NE.contexts_of(rec, n=32)
    assert info["n_clamped"] == 1 and info["n_excluded"] == 1      # context 4: its next token is padding
    assert info["ctx_index"] == [0, 1, 2, 3, 5]
    assert anc.tolist() == [3, 63, 10, 63, 5]
    assert nxt.tolist() == [int(rec["strong"]["tgt"][i, a]) for i, a in zip([0, 1, 2, 3, 5], [3, 63, 10, 63, 5])]
    assert torch.equal(tok, rec["strong"]["pos"][[0, 1, 2, 3, 5]])
    tok2, _, _, info2 = NE.contexts_of(rec, n=3)                   # more than n: the n strongest by peak
    assert info2["ctx_index"] == [0, 1, 2]
    thin = dict(rec, D_train=None)                                 # no D_train: strongest-train by peak
    assert NE.contexts_of(thin, n=4)[3]["ctx_index"] == [0, 1, 2, 3]
