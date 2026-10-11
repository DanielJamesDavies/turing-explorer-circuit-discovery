"""STANDALONE REFERENCE for DAN-36: 033's crosssae_topk.py fit, scorer and null, transcribed device-agnostic.

033 (experiments/033-cross-sae-topk/crosssae_topk.py) is the verified standalone Pythia-70m implementation. It cannot
be imported (it loads the model on CUDA at import time and keeps per-seed state in globals), so this module copies
its code path for ERROR_MODE=current, BOUND=none (the configuration that produced rows.jsonl, i.e. the 22/30 panel)
with three changes only, none of which touches the arithmetic:

  * the device is a parameter (033 hard-coded DEV="cuda");
  * the model and the dictionary cache live on a Ref033 object, the per-seed state on a Seed object, instead of
    module globals; the per-seed closures of 033's run() (natural, read, fit, score) become Seed methods;
  * TF32 ("high" float32 matmul precision, which 033 set) is enabled only on CUDA. On CPU there is no TF32.

Line-for-line correspondences are marked "033:" with the original function name. Seed selection (033's scan()) is
not re-run: the panel's windows are read from 033's scan files (see pythia_033_panel.py).

  from ref033 import Ref033
  R = Ref033("cpu"); S = R.seed("resid", 2, 15688, pos_tokens, neg_tokens)
  S.score(S.fit(True, 1e-3), "triamp400")          # 033's triamp400 arm, scored by 033's scorer
"""
import math
import random

import torch
import torch.nn.functional as F

REPO = "EleutherAI/sae-pythia-70m-32k"
MODEL = "EleutherAI/pythia-70m"
D_MODEL, D_SAE, K_SAE, N_LAYERS = 512, 32768, 16, 6
N_TRAIN = 48                     # 033: N_POS, N_TRAIN = 64, 48
STEPS, LR = 400, 0.05            # 033: STEPS, LR
LAM = 1e-3                       # 033: LAM default
TRIPLE_W, DUAL_W = 0.10, 0.25    # 033: TRIPLE_W, DUAL_W (posctx-mean term 0.10, negctx-mean term 0.25)
WEIGHT_DECAY = 0.05              # 033: AdamW(weight_decay=0.05)
THETA_INIT = 2.0                 # 033: th = torch.full(..., 2.0)


def hookpoint(site):
    """033: hookpoint."""
    kind, layer = site
    return "layers.%d" % layer if kind == "resid" else "layers.%d.%s" % (
        layer, "attention" if kind == "attn" else "mlp")


def upstream_sites(seed_layer, seed_kind="resid"):
    """033: upstream_sites. Causal prefix with the within-layer order attn < mlp < resid. NB for an mlp seed this
    includes attn(L), which Pythia's parallel residual does not route into mlp(L) (inert: zero data gradient)."""
    sites = []
    for l in range(seed_layer):
        sites += [("attn", l), ("mlp", l), ("resid", l)]
    same = {"attn": [], "mlp": [("attn", seed_layer)],
            "resid": [("attn", seed_layer), ("mlp", seed_layer)]}[seed_kind]
    return sites + same


def pre_acts(sae, x):
    return (x - sae["b_dec"]) @ sae["enc_w"].T + sae["enc_b"]


def encode(sae, x):
    """033: encode. Dense-ified Top-K code: topk_k(relu(pre_acts))."""
    p = F.relu(pre_acts(sae, x))
    vals, idx = p.topk(K_SAE, dim=-1)
    return torch.zeros_like(p).scatter(-1, idx, vals)


def seed_pre(sae, x, idx):
    return (x - sae["b_dec"]) @ sae["enc_w"][idx] + sae["enc_b"][idx]


def decode_delta(sae, delta):
    return delta @ sae["W_dec"]


class Ref033:
    """The model and the dictionaries (033's module-level `model` and `_SAE_CACHE`)."""

    def __init__(self, device="cpu"):
        from transformers import GPTNeoXForCausalLM
        self.dev = torch.device(device)
        if self.dev.type == "cuda":
            torch.set_float32_matmul_precision("high")            # 033's setting (TF32 on CUDA)
        self.model = GPTNeoXForCausalLM.from_pretrained(MODEL).float().to(self.dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._sae = {}

    def load_sae(self, site):
        """033: load_sae."""
        if site in self._sae:
            return self._sae[site]
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        sd = load_file(hf_hub_download(REPO, "%s/sae.safetensors" % hookpoint(site)), device=str(self.dev))
        sae = {"enc_w": sd["encoder.weight"], "enc_b": sd["encoder.bias"], "W_dec": sd["W_dec"], "b_dec": sd["b_dec"]}
        self._sae[site] = sae
        return sae

    def site_module(self, site):
        """033: site_module."""
        kind, layer = site
        block = self.model.gpt_neox.layers[layer]
        return {"attn": block.attention, "mlp": block.mlp, "resid": block}[kind]

    def seed(self, kind, layer, latent, pos, neg):
        return Seed(self, kind, layer, latent, pos, neg)


class Hooks:
    """033: Hooks (forward hooks at the site modules; the seed site is tapped, never edited)."""

    def __init__(self, ref, transforms, seed_site=None, seed_sae=None, seed_idx=None):
        self.ref = ref
        self.transforms = transforms
        self.seed_site, self.seed_sae, self.seed_idx = seed_site, seed_sae, seed_idx
        self.seed_out = None
        self.handles = []

    def _wrap(self, site, fn):
        def hook(mod, inp, out):
            is_tuple = isinstance(out, tuple)
            x = out[0] if is_tuple else out
            if fn is not None:
                x = fn(x)
            if site == self.seed_site:
                self.seed_out = seed_pre(self.seed_sae, x, self.seed_idx)
            if fn is None:
                return None
            return (x,) + tuple(out[1:]) if is_tuple else x
        return hook

    def __enter__(self):
        sites = set(self.transforms) | ({self.seed_site} if self.seed_site else set())
        for site in sites:
            self.handles.append(self.ref.site_module(site).register_forward_hook(
                self._wrap(site, self.transforms.get(site))))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def train_transform(saes, site, temp, params, floors):
    """033: train_transform. code = m*alpha*c + (1-m)*floor, decoded as a delta."""
    sae = saes[site]

    def fn(x):
        c = encode(sae, x)
        th, ps = params[site]
        m = torch.sigmoid(th / (temp[0] if temp else 1.0))
        chat = m * F.softplus(ps) * c
        fl = floors.get(site) if floors else None
        if fl is not None:
            chat = chat + (1 - m) * fl
        return x + decode_delta(sae, chat - c)
    return fn


def eval_transform(saes, site, members_alpha, floor, dev):
    """033: eval_transform. Members at alpha x their re-encoded value, every other latent at the floor (0 or mean)."""
    sae = saes[site]
    ma = members_alpha.get(site, {})
    idx = torch.tensor(sorted(ma), device=dev, dtype=torch.long)
    av = torch.tensor([ma[int(i)] for i in idx.tolist()], device=dev)
    fl = floor.get(site) if floor else None

    def fn(x):
        c = encode(sae, x)
        chat = fl.expand_as(c).clone() if fl is not None else torch.zeros_like(c)
        if len(idx):
            chat[..., idx] = c[..., idx] * av
        return x + decode_delta(sae, chat - c)
    return fn


class Seed:
    """033 run()'s per-seed state: splits, anchors, baselines, posctx / negctx means, pins."""

    def __init__(self, ref, kind, layer, latent, pos, neg):
        self.ref, self.dev = ref, ref.dev
        dev = self.dev
        self.kind, self.L, self.sl = kind, int(layer), int(latent)
        self.seed_site = (kind, self.L)
        self.UP = upstream_sites(self.L, kind)
        self.saes = {s: ref.load_sae(s) for s in self.UP}
        self.seed_sae = ref.load_sae(self.seed_site)
        self.pos_tr, self.pos_ho = pos[:N_TRAIN], pos[N_TRAIN:]
        self.neg_tr, self.neg_ho = neg[:N_TRAIN], neg[N_TRAIN:]

        nat_tr = self.natural(self.pos_tr)
        self.anchors_tr = nat_tr.argmax(dim=1)
        self.target_tr = nat_tr.gather(1, self.anchors_tr[:, None]).squeeze(1)
        nat_ho = self.natural(self.pos_ho)
        self.anchors_ho = nat_ho.argmax(dim=1)
        self.a_pos_ho = float(F.relu(nat_ho.gather(1, self.anchors_ho[:, None])).mean())
        neg_pre = self.natural(self.neg_ho)
        self.na_ho = neg_pre.argmax(dim=1)
        self.a_base = float(F.relu(neg_pre.gather(1, self.na_ho[:, None])).mean())

        self.means_pos, self.means_neg = self.site_means(self.pos_tr), self.site_means(self.neg_tr)
        self.pins = {}
        for site in self.UP:
            cap = {}

            def grab(mod, inp, out, _s=site):
                x = out[0] if isinstance(out, tuple) else out
                f = encode(self.saes[_s], x)
                cap[_s] = f[torch.arange(f.shape[0], device=dev), self.anchors_tr.to(dev)].mean(0)
            hd = ref.site_module(site).register_forward_hook(grab)
            with torch.no_grad():
                ref.model(self.pos_tr.to(dev), use_cache=False)
            hd.remove()
            self.pins[site] = cap[site].detach()
        self.e0 = self.read({s: eval_transform(self.saes, s, {}, {}, dev) for s in self.UP},
                            self.pos_ho, self.anchors_ho)
        self.eM = self.read({s: eval_transform(self.saes, s, {}, self.means_pos, dev) for s in self.UP},
                            self.pos_ho, self.anchors_ho)

    # ---------------------------------------------------------------------------------------- 033 run() closures
    def forward(self, tokens, hooks, grad=False):
        """033: forward (ERROR_MODE=current, no BOUND)."""
        with torch.set_grad_enabled(grad), hooks:
            self.ref.model(tokens.to(self.dev), use_cache=False)
        return hooks.seed_out

    def natural(self, t):
        return self.forward(t, Hooks(self.ref, {}, self.seed_site, self.seed_sae, self.sl)).detach()

    def anchors(self):
        """Per-context anchors of the 64 activating contexts (train then held-out) -> [64] (CPU)."""
        return torch.cat([self.anchors_tr, self.anchors_ho]).cpu()

    def site_means(self, t):
        means = {}
        for site in self.UP:
            cap = {}

            def grab(mod, inp, out, _s=site):
                x = out[0] if isinstance(out, tuple) else out
                cap[_s] = encode(self.saes[_s], x).mean(dim=(0, 1))
            hd = self.ref.site_module(site).register_forward_hook(grab)
            with torch.no_grad():
                self.ref.model(t.to(self.dev), use_cache=False)
            hd.remove()
            means[site] = cap[site].detach()
        return means

    def read(self, transforms, tokens, anchors):
        pre = self.forward(tokens, Hooks(self.ref, transforms, self.seed_site, self.seed_sae, self.sl))
        B = pre.shape[0]
        v = pre[torch.arange(B, device=self.dev), anchors.to(self.dev).clamp(0, pre.shape[1] - 1)]
        return float(F.relu(v).mean())

    def fit(self, free_amp, lam, support=None, steps=STEPS, pen_temp=True):
        """033: fit. support (for the nulls): {site: LongTensor} members fixed ON (theta +40, not trained), every
        other latent fixed OFF (-40); only the amplitudes train. pen_temp=False is NOT 033: it prices the gate as
        the production engine does (L1 and the amplitude leak guard on sigmoid(theta), not sigmoid(theta / T)),
        to isolate that one difference."""
        dev = self.dev
        params = {}
        for site in self.UP:
            th = torch.full((D_SAE,), -40.0 if support is not None else THETA_INIT, device=dev)
            if support is not None and site in support:
                th[support[site]] = 40.0
            th.requires_grad_(support is None)
            ps = torch.full((D_SAE,), math.log(math.e - 1.0), device=dev, requires_grad=free_amp)
            params[site] = (th, ps)
        opt = torch.optim.AdamW([p for pr in params.values() for p in pr if p.requires_grad], lr=LR,
                                weight_decay=WEIGHT_DECAY)
        temp = [1.0]
        tnorm = max(float((self.target_tr ** 2).mean()), 1e-6)
        floors3 = [(None, 1.0), (self.means_neg, DUAL_W), (self.means_pos, TRIPLE_W)]
        for step in range(steps):
            temp[0] = 1.0 * (0.05 ** (step / max(steps - 1, 1)))
            s0 = (step * 4) % N_TRAIN
            tk, an = self.pos_tr[s0:s0 + 4], self.anchors_tr[s0:s0 + 4]
            tg = self.target_tr[s0:s0 + 4]
            opt.zero_grad()
            for fl, w in floors3:
                tr = {s: train_transform(self.saes, s, temp, params, fl or {}) for s in self.UP}
                pre = self.forward(tk, Hooks(self.ref, tr, self.seed_site, self.seed_sae, self.sl), grad=True)
                v = pre[torch.arange(pre.shape[0], device=dev), an.to(dev)]
                (w * ((v - tg.to(dev)) ** 2).mean() / tnorm).backward()
            pen = 0.0
            for site in self.UP:
                th, ps = params[site]
                m = torch.sigmoid(th / temp[0]) if pen_temp else torch.sigmoid(th)
                if support is None:
                    pen = pen + lam * m.sum()
                if free_amp:
                    pen = pen + lam * ((1 - m) * (F.softplus(ps) - 1).abs()).sum()
            if isinstance(pen, torch.Tensor):
                pen.backward()
            opt.step()
        out = {}
        with torch.no_grad():
            for site in self.UP:
                th, ps = params[site]
                keep = (torch.sigmoid(th / temp[0]) > 0.5).nonzero(as_tuple=True)[0]
                al = F.softplus(ps)[keep] if free_amp else torch.ones(len(keep), device=dev)
                out[site] = {int(i): float(a) for i, a in zip(keep.tolist(), al.tolist())}
        return out

    def score(self, ma, tag):
        """033: score -> its row (ampF0, ampFM, sup, cf_amp, pin0, pinM on the held-out 16). `ma` is
        {(kind, layer): {latent: alpha}}; sites outside self.UP are ignored (033 never had any)."""
        dev, UP, saes = self.dev, self.UP, self.saes
        a_pos_ho, e0, eM, a_base = self.a_pos_ho, self.e0, self.eM, self.a_base
        ma = {s: d for s, d in ma.items() if s in UP and d}
        n = sum(len(d) for d in ma.values())
        aw0 = self.read({s: eval_transform(saes, s, ma, {}, dev) for s in UP}, self.pos_ho, self.anchors_ho)
        awM = self.read({s: eval_transform(saes, s, ma, self.means_pos, dev) for s in UP},
                        self.pos_ho, self.anchors_ho)
        sup_tr = {}
        for site in UP:
            d = ma.get(site, {})
            if not d:
                continue
            idx = torch.tensor(sorted(d), device=dev, dtype=torch.long)

            def fn(x, _sae=saes[site], _idx=idx):
                c = encode(_sae, x)
                chat = c.clone()
                chat[..., _idx] = 0.0
                return x + decode_delta(_sae, chat - c)
            sup_tr[site] = fn
        a_sup = self.read(sup_tr, self.pos_ho, self.anchors_ho) if sup_tr else a_pos_ho
        inj = {}
        for site in UP:
            d = ma.get(site, {})
            if not d:
                continue
            idx = torch.tensor(sorted(d), device=dev, dtype=torch.long)
            vals = torch.tensor([d[int(i)] * float(self.pins[site][int(i)]) for i in idx.tolist()], device=dev)

            def fn(x, _sae=saes[site], _idx=idx, _v=vals):
                c = encode(_sae, x)
                chat = c.clone()
                chat[..., _idx] = _v
                return x + decode_delta(_sae, chat - c)
            inj[site] = fn
        a_inj = self.read(inj, self.neg_ho, self.na_ho) if inj else a_base

        def pin_read(floor):
            cap_, hs_ = {}, []
            for site_ in UP:
                d_ = ma.get(site_, {})
                if not d_:
                    continue
                ix = torch.tensor(sorted(d_), device=dev, dtype=torch.long)

                def grab(mod, inp, out, _s=site_, _ix=ix):
                    x = out[0] if isinstance(out, tuple) else out
                    cap_[_s] = encode(saes[_s], x)[..., _ix].detach()
                hs_.append(self.ref.site_module(site_).register_forward_hook(grab))
            try:
                with torch.no_grad():
                    self.ref.model(self.pos_ho.to(dev), use_cache=False)
            finally:
                for h_ in hs_:
                    h_.remove()
            tr_ = {}
            for site_ in UP:
                d_ = ma.get(site_, {})
                ix = torch.tensor(sorted(d_), device=dev, dtype=torch.long) if d_ else None
                av = torch.tensor([d_[int(i)] for i in ix.tolist()], device=dev) if d_ else None
                fl_ = floor.get(site_) if floor else None

                def fn(x, _s=site_, _ix=ix, _av=av, _fl=fl_):
                    c = encode(saes[_s], x)
                    chat = _fl.expand_as(c).clone() if _fl is not None else torch.zeros_like(c)
                    if _ix is not None:
                        chat[..., _ix] = cap_[_s] * _av
                    return x + decode_delta(saes[_s], chat - c)
                tr_[site_] = fn
            return self.read(tr_, self.pos_ho, self.anchors_ho)
        a_pin0 = pin_read({}) if n else e0
        a_pinM = pin_read(self.means_pos) if n else eM
        r4 = lambda v: round(v, 4)  # noqa: E731
        return {"layer": self.L, "kind": self.kind, "latent": self.sl, "arm": tag, "n": n,
                "ampF0": r4((aw0 - e0) / (a_pos_ho - e0)) if abs(a_pos_ho - e0) > 1e-9 else None,
                "ampFM": r4((awM - eM) / (a_pos_ho - eM)) if abs(a_pos_ho - eM) > 1e-9 else None,
                "sup": r4(1.0 - a_sup / a_pos_ho) if a_pos_ho > 1e-9 else None,
                "cf_amp": r4((a_inj - a_base) / (a_pos_ho - a_base)) if abs(a_pos_ho - a_base) > 1e-9 else None,
                "pin0": r4((a_pin0 - e0) / (a_pos_ho - e0)) if abs(a_pos_ho - e0) > 1e-9 else None,
                "pinM": r4((a_pinM - eM) / (a_pos_ho - eM)) if abs(a_pos_ho - eM) > 1e-9 else None,
                "a_pos_ho": round(a_pos_ho, 3)}

    def live_pool(self):
        """033: the live pool (every upstream latent that fires anywhere on the training contexts) and the anchor
        support rate; nulls draw from it."""
        dev, live, anchor_rate = self.dev, [], []
        for site in self.UP:
            cap = {}

            def grab(mod, inp, out, _s=site):
                x = out[0] if isinstance(out, tuple) else out
                f = encode(self.saes[_s], x)
                cap["live"] = (f > 0).reshape(-1, D_SAE).any(0)
                cap["anchor"] = (f[torch.arange(f.shape[0], device=dev), self.anchors_tr.to(dev)] > 0).float()
            hd = self.ref.site_module(site).register_forward_hook(grab)
            with torch.no_grad():
                self.ref.model(self.pos_tr.to(dev), use_cache=False)
            hd.remove()
            lm = cap["live"]
            live += [(site, int(i)) for i in lm.nonzero(as_tuple=True)[0].tolist()]
            if lm.any():
                anchor_rate.append(float(cap["anchor"][:, lm].mean()))
        return live, sum(anchor_rate) / max(len(anchor_rate), 1)

    def null_members(self, live, n, draw, n_draws):
        """033's draws: rng = Random(1000 + latent), `n_draws` successive samples of size n; returns draw `draw`'s
        support {site: LongTensor}. (033 sampled at its own circuit's n; pass the n you are matching.)"""
        rng = random.Random(1000 + self.sl)
        members = None
        for d in range(n_draws):
            members = rng.sample(live, min(n, len(live)))
            if d == draw:
                break
        support = {}
        for s, i in members:
            support.setdefault(s, []).append(i)
        return {s: torch.tensor(v, dtype=torch.long, device=self.dev) for s, v in support.items()}
