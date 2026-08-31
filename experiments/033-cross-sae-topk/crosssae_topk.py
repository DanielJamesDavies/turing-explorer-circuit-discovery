"""Cross-SAE replication on GENUINE Top-K dictionaries.

Model:  EleutherAI/pythia-70m (the model these SAEs were trained on)
SAEs:   EleutherAI/sae-pythia-70m-32k — trained Top-K, k=16, 32,768
        latents, d_in 512, decoder-normalised; resid / attention / mlp
        hookpoints for all 6 layers (18 dictionaries), 8.2B Pile tokens.
Corpus: wikitext-103 windows (as in the dense run).

This supersedes the dense-ReLU arm as THE cross-SAE replication: the
method's pre-activation objective is motivated by Top-K censoring, so
Top-K dictionaries are its domain. Nothing here simulates sparsity —
the codes are k-sparse because the encoder was trained that way.

Semantics transcribed from EleutherAI/sparsify:
    pre_acts = (x - b_dec) @ W_enc.T + b_enc      (seed read: raw)
    code     = topk_k(relu(pre_acts))             (upstream sites)
    decode   = code @ W_dec + b_dec               (b_dec cancels in the
                                                   delta intervention)

Arms per seed: triamp400 (triple floor + free amplitudes), gate400,
and N_NULL amplitude-fitted random sets drawn from the live pool.
48/16 held-out split; negatives VERIFIED inactive.

REGISTERED PREDICTION (2026-08-11, before running): discovered
circuits stay faithful and necessary, and fitted-random nulls fail at
EVERY n — restoring the home behaviour on a public model, because a
random latent sits outside the top-k at the anchor ~99.95% of the time
(k=16 of 32,768) and its amplitude multiplies exactly zero.

  PYTHONPATH=. python crosssae_topk.py scan
  PYTHONPATH=. python crosssae_topk.py run
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, GPTNeoXForCausalLM

HERE = Path(__file__).parent
REPO = "EleutherAI/sae-pythia-70m-32k"
MODEL = "EleutherAI/pythia-70m"
DEV = "cuda"
D_MODEL, D_SAE, K_SAE, N_LAYERS = 512, 32768, 16, 6
SEQ_LEN, BATCH = 64, 32
N_POS, N_TRAIN = 64, 48
SEED_LAYERS = [int(x) for x in os.environ.get("SEED_LAYERS", "2,4").split(",")]
SEED_KINDS = os.environ.get("SEED_KINDS", "resid").split(",")
SEEDS_PER_LAYER = int(os.environ.get("SEEDS_PER_CELL", 3))
SCAN_FILE = os.environ.get("SCAN_FILE", "scan.pt")
N_NULL = int(os.environ.get("N_NULL", 3))
STEPS, LR = 400, 0.05
LAM = float(os.environ.get("LAM", 1e-3))
TAG = "" if LAM == 1e-3 else "_lam%g" % LAM
TRIPLE_W, DUAL_W = 0.10, 0.25
torch.set_float32_matmul_precision("high")

tok = AutoTokenizer.from_pretrained(MODEL)
model = GPTNeoXForCausalLM.from_pretrained(MODEL).float().to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)

_SAE_CACHE = {}


def load_sae(hookpoint):
    if hookpoint in _SAE_CACHE:
        return _SAE_CACHE[hookpoint]
    from safetensors.torch import load_file
    path = hf_hub_download(REPO, "%s/sae.safetensors" % hookpoint)
    sd = load_file(path, device=DEV)
    sae = {"enc_w": sd["encoder.weight"], "enc_b": sd["encoder.bias"],
           "W_dec": sd["W_dec"], "b_dec": sd["b_dec"]}
    _SAE_CACHE[hookpoint] = sae
    return sae


def pre_acts(sae, x):
    return (x - sae["b_dec"]) @ sae["enc_w"].T + sae["enc_b"]


def encode(sae, x):
    """Dense-ified Top-K code: topk_k(relu(pre_acts))."""
    p = F.relu(pre_acts(sae, x))
    vals, idx = p.topk(K_SAE, dim=-1)
    return torch.zeros_like(p).scatter(-1, idx, vals)


def seed_pre(sae, x, idx):
    return (x - sae["b_dec"]) @ sae["enc_w"][idx] + sae["enc_b"][idx]


def decode_delta(sae, delta):
    return delta @ sae["W_dec"]


def hookpoint(site):
    kind, layer = site
    return "layers.%d" % layer if kind == "resid" else "layers.%d.%s" % (
        layer, "attention" if kind == "attn" else "mlp")


def site_module(site):
    kind, layer = site
    block = model.gpt_neox.layers[layer]
    return {"attn": block.attention, "mlp": block.mlp, "resid": block}[kind]


def upstream_sites(seed_layer, seed_kind="resid"):
    """Causal prefix, with the within-layer order attn < mlp < resid."""
    sites = []
    for l in range(seed_layer):
        sites += [("attn", l), ("mlp", l), ("resid", l)]
    same = {"attn": [], "mlp": [("attn", seed_layer)],
            "resid": [("attn", seed_layer), ("mlp", seed_layer)]}[seed_kind]
    return sites + same


class Hooks:
    def __init__(self, transforms, seed_site=None, seed_sae=None,
                 seed_idx=None):
        self.transforms = transforms
        self.seed_site, self.seed_sae, self.seed_idx = (seed_site, seed_sae,
                                                        seed_idx)
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
        sites = set(self.transforms) | ({self.seed_site}
                                        if self.seed_site else set())
        for site in sites:
            self.handles.append(site_module(site).register_forward_hook(
                self._wrap(site, self.transforms.get(site))))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def forward(tokens, hooks, grad=False):
    with torch.set_grad_enabled(grad), hooks:
        model(tokens.to(DEV), use_cache=False)
    return hooks.seed_out


def train_transform(saes, site, temp, params, floors):
    """code = m*alpha*c + (1-m)*floor, decoded as a delta."""
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


def eval_transform(saes, site, members_alpha, floor):
    sae = saes[site]
    ma = members_alpha.get(site, {})
    idx = torch.tensor(sorted(ma), device=DEV, dtype=torch.long)
    av = torch.tensor([ma[int(i)] for i in idx.tolist()], device=DEV)
    fl = floor.get(site) if floor else None
    def fn(x):
        c = encode(sae, x)
        chat = (fl.expand_as(c).clone() if fl is not None
                else torch.zeros_like(c))
        if len(idx):
            chat[..., idx] = c[..., idx] * av
        return x + decode_delta(sae, chat - c)
    return fn


def windows(limit):
    import pyarrow.parquet as pq
    path = hf_hub_download(
        repo_id="Salesforce/wikitext", repo_type="dataset",
        filename="wikitext-103-raw-v1/train-00000-of-00002.parquet")
    texts = pq.read_table(path, columns=["text"])["text"].to_pylist()
    buf, out = [], []
    for t_ in texts:
        if not t_.strip():
            continue
        buf.extend(tok(t_, add_special_tokens=False)["input_ids"])
        while len(buf) >= SEQ_LEN:
            out.append(buf[:SEQ_LEN]); buf = buf[SEQ_LEN:]
            if len(out) >= limit:
                return torch.tensor(out, dtype=torch.long)
    return torch.tensor(out, dtype=torch.long)


def scan():
    toks = windows(20000)
    print("windows:", tuple(toks.shape), flush=True)
    stats = {}
    for L in SEED_LAYERS:
        for kind in SEED_KINDS:
            site = (kind, L)
            if kind != "resid" and L == 0:
                continue
            sae = load_sae(hookpoint(site))
            fire = torch.zeros(D_SAE, device=DEV)
            mx = torch.zeros(D_SAE, device=DEV)
            chunks = []
            with torch.no_grad():
                for s0 in range(0, len(toks), BATCH):
                    cap = {}
                    def grab(m, i, o):
                        cap["x"] = o[0] if isinstance(o, tuple) else o
                    hd = site_module(site).register_forward_hook(grab)
                    model(toks[s0:s0 + BATCH].to(DEV), use_cache=False)
                    hd.remove()
                    wmax = encode(sae, cap["x"]).amax(dim=1)
                    fire += (wmax > 0).float().sum(0)
                    mx = torch.maximum(mx, wmax.amax(0))
                    chunks.append(wmax.half().cpu())
            frac = fire / len(toks)
            band = ((frac > 0.005) & (frac < 0.05)).nonzero(as_tuple=True)[0]
            if not len(band):
                print("L%d %s: no seeds in band" % (L, kind), flush=True)
                del chunks
                continue
            chosen = band[mx[band].argsort(descending=True)[:SEEDS_PER_LAYER]]
            wmax_all = torch.cat(chunks)
            for sl in chosen.tolist():
                col = wmax_all[:, sl]
                top = col.argsort(descending=True)[:N_POS].tolist()
                silent = (col == 0).nonzero(as_tuple=True)[0]
                neg = silent[torch.randperm(len(silent))[:N_POS]].tolist()
                stats["%s/%d/%d" % (kind, L, sl)] = {
                    "layer": L, "kind": kind, "latent": int(sl),
                    "fire_frac": round(float(frac[sl]), 4),
                    "max_act": round(float(mx[sl]), 3),
                    "pos_windows": top, "neg_windows": neg}
                print("L%d %-5s latent %d: frac %.3f max %.2f"
                      % (L, kind, sl, frac[sl], mx[sl]), flush=True)
            del wmax_all, chunks
    torch.save({"tokens": toks, "seeds": stats}, HERE / SCAN_FILE)
    print("SCAN DONE (%d seeds)" % len(stats), flush=True)


def run():
    data = torch.load(HERE / SCAN_FILE, weights_only=False)
    toks, seeds = data["tokens"], data["seeds"]
    rows_path = HERE / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.open():
            try:
                r = json.loads(line)
                done.add((r.get("kind", "resid"), r["layer"], r["latent"],
                          r["arm"]))
            except Exception:
                pass
    fh = rows_path.open("a")

    for key, S in seeds.items():
        L, sl = S["layer"], S["latent"]
        kind = S.get("kind", "resid")
        seed_site = (kind, L)
        if (kind, L, sl, "null%d" % (N_NULL - 1) + TAG) in done:
            print("[L%d %s %d] complete, skipping" % (L, kind, sl),
                  flush=True)
            continue
        UP = upstream_sites(L, kind)
        if not UP:
            continue
        saes = {s: load_sae(hookpoint(s)) for s in UP}
        seed_sae = load_sae(hookpoint(seed_site))
        pos, neg = toks[S["pos_windows"]], toks[S["neg_windows"]]
        pos_tr, pos_ho = pos[:N_TRAIN], pos[N_TRAIN:]
        neg_tr, neg_ho = neg[:N_TRAIN], neg[N_TRAIN:]

        def natural(t):
            return forward(t, Hooks({}, seed_site, seed_sae, sl)).detach()

        nat_tr = natural(pos_tr)
        anchors_tr = nat_tr.argmax(dim=1)
        target_tr = nat_tr.gather(1, anchors_tr[:, None]).squeeze(1)
        nat_ho = natural(pos_ho)
        anchors_ho = nat_ho.argmax(dim=1)
        a_pos_ho = float(F.relu(nat_ho.gather(1, anchors_ho[:, None])).mean())
        neg_pre = natural(neg_ho)
        na_ho = neg_pre.argmax(dim=1)
        a_base = float(F.relu(neg_pre.gather(1, na_ho[:, None])).mean())
        if a_pos_ho < 0.05:
            print("[L%d %s %d] held-out a_pos too small, skip" % (L, kind, sl),
                  flush=True)
            continue

        def site_means(t):
            means = {}
            for site in UP:
                cap = {}
                def grab(mod, inp, out, _s=site):
                    x = out[0] if isinstance(out, tuple) else out
                    cap[_s] = encode(saes[_s], x).mean(dim=(0, 1))
                hd = site_module(site).register_forward_hook(grab)
                with torch.no_grad():
                    model(t.to(DEV), use_cache=False)
                hd.remove()
                means[site] = cap[site].detach()
            return means

        means_pos, means_neg = site_means(pos_tr), site_means(neg_tr)
        pins = {}
        for site in UP:
            cap = {}
            def grab(mod, inp, out, _s=site):
                x = out[0] if isinstance(out, tuple) else out
                f = encode(saes[_s], x)
                cap[_s] = f[torch.arange(f.shape[0], device=DEV),
                            anchors_tr.to(DEV)].mean(0)
            hd = site_module(site).register_forward_hook(grab)
            with torch.no_grad():
                model(pos_tr.to(DEV), use_cache=False)
            hd.remove()
            pins[site] = cap[site].detach()

        def read(transforms, tokens, anchors):
            pre = forward(tokens, Hooks(transforms, seed_site, seed_sae, sl))
            B = pre.shape[0]
            v = pre[torch.arange(B, device=DEV),
                    anchors.to(DEV).clamp(0, pre.shape[1] - 1)]
            return float(F.relu(v).mean())

        e0 = read({s: eval_transform(saes, s, {}, {}) for s in UP},
                  pos_ho, anchors_ho)
        eM = read({s: eval_transform(saes, s, {}, means_pos) for s in UP},
                  pos_ho, anchors_ho)
        print("\n[L%d %d] a_pos_ho %.3f | e0 %.3f eM %.3f | a_base %.3f"
              % (L, sl, a_pos_ho, e0, eM, a_base), flush=True)

        def fit(free_amp, lam, support=None):
            params = {}
            for site in UP:
                th = torch.full((D_SAE,), -40.0 if support is not None
                                else 2.0, device=DEV)
                if support is not None and site in support:
                    th[support[site]] = 40.0
                th.requires_grad_(support is None)
                ps = torch.full((D_SAE,), math.log(math.e - 1.0), device=DEV,
                                requires_grad=free_amp)
                params[site] = (th, ps)
            opt = torch.optim.AdamW([p for pr in params.values() for p in pr
                                     if p.requires_grad], lr=LR,
                                    weight_decay=0.05)
            temp = [1.0]
            tnorm = max(float((target_tr ** 2).mean()), 1e-6)
            floors3 = [(None, 1.0), (means_neg, DUAL_W), (means_pos, TRIPLE_W)]
            for step in range(STEPS):
                temp[0] = 1.0 * (0.05 ** (step / max(STEPS - 1, 1)))
                s0 = (step * 4) % N_TRAIN
                tk, an = pos_tr[s0:s0 + 4], anchors_tr[s0:s0 + 4]
                tg = target_tr[s0:s0 + 4]
                opt.zero_grad()
                for fl, w in floors3:
                    tr = {s: train_transform(saes, s, temp, params, fl or {})
                          for s in UP}
                    pre = forward(tk, Hooks(tr, seed_site, seed_sae, sl), grad=True)
                    v = pre[torch.arange(pre.shape[0], device=DEV), an.to(DEV)]
                    (w * ((v - tg.to(DEV)) ** 2).mean() / tnorm).backward()
                pen = 0.0
                for site in UP:
                    th, ps = params[site]
                    m = torch.sigmoid(th / temp[0])
                    if support is None:
                        pen = pen + lam * m.sum()
                    if free_amp:
                        pen = pen + lam * ((1 - m)
                                           * (F.softplus(ps) - 1).abs()).sum()
                if isinstance(pen, torch.Tensor):
                    pen.backward()
                opt.step()
            out = {}
            with torch.no_grad():
                for site in UP:
                    th, ps = params[site]
                    keep = (torch.sigmoid(th / temp[0]) > 0.5).nonzero(
                        as_tuple=True)[0]
                    al = (F.softplus(ps)[keep] if free_amp
                          else torch.ones(len(keep), device=DEV))
                    out[site] = {int(i): float(a) for i, a in
                                 zip(keep.tolist(), al.tolist())}
            return out

        def score(ma, tag, secs):
            n = sum(len(d) for d in ma.values())
            aw0 = read({s: eval_transform(saes, s, ma, {}) for s in UP},
                       pos_ho, anchors_ho)
            awM = read({s: eval_transform(saes, s, ma, means_pos)
                        for s in UP}, pos_ho, anchors_ho)
            sup_tr = {}
            for site in UP:
                d = ma.get(site, {})
                if not d:
                    continue
                sae = saes[site]
                idx = torch.tensor(sorted(d), device=DEV, dtype=torch.long)
                def fn(x, _sae=sae, _idx=idx):
                    c = encode(_sae, x)
                    chat = c.clone()
                    chat[..., _idx] = 0.0
                    return x + decode_delta(_sae, chat - c)
                sup_tr[site] = fn
            a_sup = read(sup_tr, pos_ho, anchors_ho) if sup_tr else a_pos_ho
            inj = {}
            for site in UP:
                d = ma.get(site, {})
                if not d:
                    continue
                sae = saes[site]
                idx = torch.tensor(sorted(d), device=DEV, dtype=torch.long)
                vals = torch.tensor([d[int(i)] * float(pins[site][int(i)])
                                     for i in idx.tolist()], device=DEV)
                def fn(x, _sae=sae, _idx=idx, _v=vals):
                    c = encode(_sae, x)
                    chat = c.clone()
                    chat[..., _idx] = _v
                    return x + decode_delta(_sae, chat - c)
                inj[site] = fn
            a_inj = read(inj, neg_ho, na_ho) if inj else a_base
            row = {"layer": L, "kind": kind, "latent": sl, "arm": tag,
                   "n": n,
                   "ampF0": round((aw0 - e0) / (a_pos_ho - e0), 4)
                   if abs(a_pos_ho - e0) > 1e-9 else None,
                   "ampFM": round((awM - eM) / (a_pos_ho - eM), 4)
                   if abs(a_pos_ho - eM) > 1e-9 else None,
                   "sup": round(1.0 - a_sup / a_pos_ho, 4)
                   if a_pos_ho > 1e-9 else None,
                   "cf_amp": round((a_inj - a_base) / (a_pos_ho - a_base), 4)
                   if abs(a_pos_ho - a_base) > 1e-9 else None,
                   "a_pos_ho": round(a_pos_ho, 3), "secs": round(secs, 1)}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("  %-14s n=%-6d ampF0=%-8s ampFM=%-8s sup=%-7s cf_amp=%s"
                  % (tag, n, row["ampF0"], row["ampFM"], row["sup"],
                     row["cf_amp"]), flush=True)
            return row

        n_ref = None
        if (kind, L, sl, "triamp400" + TAG) not in done:
            t0 = time.time()
            n_ref = score(fit(True, LAM), "triamp400" + TAG,
                          time.time() - t0)["n"]
        if (kind, L, sl, "gate400" + TAG) not in done:
            t0 = time.time()
            score(fit(False, LAM), "gate400" + TAG, time.time() - t0)
        if n_ref is None:
            for line in rows_path.open():
                r = json.loads(line)
                if (r.get("kind", "resid"), r["layer"], r["latent"],
                        r["arm"]) == (kind, L, sl, "triamp400" + TAG):
                    n_ref = r["n"]
        # live pool + ANCHOR SUPPORT (the mechanism statistic, measured on
        # genuine Top-K codes for direct comparison with the dense run)
        live, anchor_rate = [], []
        for site in UP:
            cap = {}
            def grab(mod, inp, out, _s=site):
                x = out[0] if isinstance(out, tuple) else out
                f = encode(saes[_s], x)
                cap["live"] = (f > 0).reshape(-1, D_SAE).any(0)
                cap["anchor"] = (f[torch.arange(f.shape[0], device=DEV),
                                   anchors_tr.to(DEV)] > 0).float()
            hd = site_module(site).register_forward_hook(grab)
            with torch.no_grad():
                model(pos_tr.to(DEV), use_cache=False)
            hd.remove()
            lm = cap["live"]
            live += [(site, int(i)) for i in lm.nonzero(as_tuple=True)[0]
                     .tolist()]
            if lm.any():
                anchor_rate.append(float(cap["anchor"][:, lm].mean()))
        rate = sum(anchor_rate) / max(len(anchor_rate), 1)
        fh.write(json.dumps({"layer": L, "kind": kind, "latent": sl,
                             "arm": "anchor_support",
                             "n": len(live), "anchor_support_rate":
                             round(rate, 5)}) + "\n"); fh.flush()
        print("  anchor support %.5f over live pool %d (k=%d/%d)"
              % (rate, len(live), K_SAE, D_SAE), flush=True)
        if n_ref:
            rng = random.Random(1000 + sl)
            for draw in range(N_NULL):
                members = rng.sample(live, min(n_ref, len(live)))
                tag = "null%d" % draw + TAG
                if (kind, L, sl, tag) in done:
                    continue
                support = {}
                for s, i in members:
                    support.setdefault(s, []).append(i)
                support = {s: torch.tensor(v, dtype=torch.long, device=DEV)
                           for s, v in support.items()}
                t0 = time.time()
                score(fit(True, 0.0, support=support), tag, time.time() - t0)
        del saes, seed_sae
        torch.cuda.empty_cache()
    fh.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    {"scan": scan, "run": run}[sys.argv[1]]()
