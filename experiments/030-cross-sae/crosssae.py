"""Cross-SAE replication: the weighted-circuit core claims on
Pythia-70m-deduped with Marks et al.'s public ReLU dictionaries — a
different model, a different corpus, and a different SAE architecture
(ReLU+L1 dense, not Top-K) from everything in the paper's evidence.

Claims under test (the panel's headline set):
  1. tri-amp (triple floor + free amplitudes) yields compact weighted
     circuits faithful under BOTH the zero and mean fills, held-out;
  2. gate-only masks fail the same exam at larger sizes;
  3. random same-size sets with identically fitted amplitudes fail;
  4. necessity (suppression) holds; amplitudes drive on held-out
     negative contexts.

Everything is a purpose-built ~self-contained harness (no import of
the TuringLLM pipeline): hooks on GPTNeoX submodule outputs, the
intervention x + decoder(c_hat - c) (bias cancels, reconstruction
error passes through), seed read = encoder pre-activation at the seed
site. Negatives are VERIFIED inactive (zero firing in the window) —
stronger than the home pipeline's unverified retrieval.

Stages:
  SCAN   pick seeds per resid layer (fire-rate in band, strong max)
         and collect top-64 posctx windows + 64 verified-silent negctx
  RUN    per seed: triamp400 / gate400 arms + N_NULL fitted nulls,
         48/16 held-out split, ampF0/ampFMd/sup/cf_amp

  PYTHONPATH=. python crosssae.py scan
  PYTHONPATH=. python crosssae.py run
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPTNeoXForCausalLM

HERE = Path(__file__).parent
DICT_ROOT = HERE / "dictionaries" / "pythia-70m-deduped"
DEV = "cuda"
D_MODEL, D_SAE, N_LAYERS = 512, 32768, 6
SEQ_LEN, BATCH = 64, 32
N_POS, N_TRAIN = 64, 48
SEED_LAYERS = [2, 4]          # resid seeds, shallow-ish and deep-ish
SEEDS_PER_LAYER = 3
N_NULL = 3
import os
STEPS, LR = 400, 0.05
LAM = float(os.environ.get("LAM", 1e-3))
TAG_SUFFIX = "" if LAM == 1e-3 else "_lam%g" % LAM
TRIPLE_W, DUAL_W = 0.10, 0.25
torch.set_float32_matmul_precision("high")

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m-deduped").float().to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)


def load_sae(sub):
    sd = torch.load(DICT_ROOT / sub / "10_32768" / "ae.pt",
                    map_location=DEV, weights_only=True)
    return {"bias": sd["bias"], "enc_w": sd["encoder.weight"],
            "enc_b": sd["encoder.bias"], "dec_w": sd["decoder.weight"]}


def encode(sae, x):
    return F.relu((x - sae["bias"]) @ sae["enc_w"].T + sae["enc_b"])


def seed_pre(sae, x, idx):
    return (x - sae["bias"]) @ sae["enc_w"][idx] + sae["enc_b"][idx]


def site_module(site):
    kind, layer = site
    if kind == "embed":
        return model.gpt_neox.embed_in
    block = model.gpt_neox.layers[layer]
    return {"attn": block.attention, "mlp": block.mlp, "resid": block}[kind]


def upstream_sites(seed_layer):
    sites = [("embed", -1)]
    for l in range(seed_layer):
        sites += [("attn", l), ("mlp", l), ("resid", l)]
    sites += [("attn", seed_layer), ("mlp", seed_layer)]
    return sites


def sub_name(site):
    kind, layer = site
    if kind == "embed":
        return "embed"
    return "%s_out_layer%d" % (kind, layer)


class Hooks:
    """Register transforms on submodule outputs + a seed reader."""

    def __init__(self, transforms, seed_site=None, seed_sae=None,
                 seed_idx=None):
        self.transforms = transforms          # {site: fn(x)->x}
        self.seed_site, self.seed_sae = seed_site, seed_sae
        self.seed_idx = seed_idx
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
            fn = self.transforms.get(site)
            self.handles.append(
                site_module(site).register_forward_hook(self._wrap(site, fn)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def forward(tokens, hooks, grad=False):
    with torch.set_grad_enabled(grad), hooks:
        model(tokens.to(DEV), use_cache=False)
    return hooks.seed_out


def spec_transform_factory(saes, site, temp, params, floors):
    """Training transform: code = m*alpha*c + (1-m)*floor. Module-level so
    the mechanism test (topkeval.py) trains with identical semantics."""
    sae = saes[site]
    def fn(x):
        c = encode(sae, x)
        th, ps = params[site]
        m = torch.sigmoid(th / (temp[0] if temp else 1.0))
        alpha = F.softplus(ps)
        fl = floors.get(site) if floors else None
        chat = m * alpha * c
        if fl is not None:
            chat = chat + (1 - m) * fl
        return x + (chat - c) @ sae["dec_w"].T
    return fn


# ----------------------------------------------------------------- corpus
def windows(limit):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
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
            out.append(buf[:SEQ_LEN])
            buf = buf[SEQ_LEN:]
            if len(out) >= limit:
                return torch.tensor(out, dtype=torch.long)
    return torch.tensor(out, dtype=torch.long)


def scan():
    n_windows = 20000
    toks = windows(n_windows)
    print("windows:", toks.shape, flush=True)
    stats = {}
    for L in SEED_LAYERS:
        sae = load_sae(sub_name(("resid", L)))
        fire = torch.zeros(D_SAE, device=DEV)
        mx = torch.zeros(D_SAE, device=DEV)
        acts_max = []            # per-window max act per CHOSEN latents later
        with torch.no_grad():
            for s0 in range(0, len(toks), BATCH):
                h = Hooks({}, ("resid", L), sae, 0)  # reuse reader for x
                # capture the block output directly
                cap = {}
                def grab(mod, inp, out):
                    cap["x"] = out[0]
                hd = site_module(("resid", L)).register_forward_hook(grab)
                model(toks[s0:s0 + BATCH].to(DEV), use_cache=False)
                hd.remove()
                f = encode(sae, cap["x"])            # [B,T,D_SAE]
                wmax = f.amax(dim=1)                 # [B,D_SAE]
                fire += (wmax > 0).float().sum(0)
                mx = torch.maximum(mx, wmax.amax(0))
                acts_max.append(wmax.half().cpu())
        frac = fire / len(toks)
        band = ((frac > 0.005) & (frac < 0.05)).nonzero(as_tuple=True)[0]
        order = mx[band].argsort(descending=True)
        chosen = band[order[:SEEDS_PER_LAYER]].tolist()
        wmax_all = torch.cat(acts_max)               # [N,D_SAE] cpu
        for sl in chosen:
            col = wmax_all[:, sl]
            top = col.argsort(descending=True)[:N_POS].tolist()
            silent = (col == 0).nonzero(as_tuple=True)[0]
            neg = silent[torch.randperm(len(silent))[:N_POS]].tolist()
            stats["%d/%d" % (L, sl)] = {
                "layer": L, "latent": int(sl),
                "fire_frac": round(float(frac[sl]), 4),
                "max_act": round(float(mx[sl]), 3),
                "pos_windows": top, "neg_windows": neg}
            print("L%d latent %d: frac %.3f max %.2f" %
                  (L, sl, frac[sl], mx[sl]), flush=True)
        del wmax_all, acts_max
    torch.save({"tokens": toks, "seeds": stats}, HERE / "scan.pt")
    print("SCAN DONE", flush=True)


# ------------------------------------------------------------------- run
def run():
    data = torch.load(HERE / "scan.pt", weights_only=False)
    toks, seeds = data["tokens"], data["seeds"]
    fh = (HERE / "rows.jsonl").open("a")
    done = set()
    if (HERE / "rows.jsonl").exists():
        for line in (HERE / "rows.jsonl").open():
            try:
                r = json.loads(line)
                done.add((r["layer"], r["latent"], r["arm"]))
            except Exception:
                pass

    for key, S in seeds.items():
        L, sl = S["layer"], S["latent"]
        if (L, sl, "null%d" % (N_NULL - 1) + TAG_SUFFIX) in done:
            print("[L%d %d] complete, skipping" % (L, sl), flush=True)
            continue
        UP = upstream_sites(L)
        saes = {site: load_sae(sub_name(site)) for site in UP}
        seed_sae = load_sae(sub_name(("resid", L)))
        pos = toks[S["pos_windows"]]
        neg = toks[S["neg_windows"]]
        pos_tr, pos_ho = pos[:N_TRAIN], pos[N_TRAIN:]
        neg_tr, neg_ho = neg[:N_TRAIN], neg[N_TRAIN:]

        def natural(tokens):
            h = Hooks({}, ("resid", L), seed_sae, sl)
            return forward(tokens, h).detach()

        nat_tr = natural(pos_tr)
        anchors_tr = nat_tr.argmax(dim=1)
        target_tr = nat_tr.gather(1, anchors_tr[:, None]).squeeze(1)
        nat_ho = natural(pos_ho)
        anchors_ho = nat_ho.argmax(dim=1)
        a_pos_ho = float(F.relu(nat_ho.gather(
            1, anchors_ho[:, None])).mean())
        neg_pre_ho = natural(neg_ho)
        na_ho = neg_pre_ho.argmax(dim=1)
        a_base = float(F.relu(neg_pre_ho.gather(1, na_ho[:, None])).mean())

        # floors + pins from TRAIN
        def site_means(tokens):
            means = {}
            for site in UP:
                cap = {}
                def grab(mod, inp, out, _s=site):
                    x = out[0] if isinstance(out, tuple) else out
                    cap[_s] = encode(saes[_s], x).mean(dim=(0, 1))
                hd = site_module(site).register_forward_hook(grab)
                with torch.no_grad():
                    model(tokens.to(DEV), use_cache=False)
                hd.remove()
                means[site] = cap[site].detach()
            return means

        means_pos = site_means(pos_tr)
        means_neg = site_means(neg_tr)
        pins = {}
        for site in UP:
            cap = {}
            def grab(mod, inp, out, _s=site):
                x = out[0] if isinstance(out, tuple) else out
                f = encode(saes[_s], x)
                B = f.shape[0]
                cap[_s] = f[torch.arange(B, device=DEV),
                            anchors_tr.to(DEV)].mean(0)
            hd = site_module(site).register_forward_hook(grab)
            with torch.no_grad():
                model(pos_tr.to(DEV), use_cache=False)
            hd.remove()
            pins[site] = cap[site].detach()

        def spec_transform(site, kind_spec, temp=None, params=None,
                           floors=None):
            """Training transform: code = m*alpha*c + (1-m)*floor."""
            sae = saes[site]
            def fn(x):
                c = encode(sae, x)
                th, ps = params[site]
                m = torch.sigmoid(th / (temp[0] if temp else 1.0))
                alpha = F.softplus(ps)
                fl = floors.get(site) if floors else None
                chat = m * alpha * c
                if fl is not None:
                    chat = chat + (1 - m) * fl
                return x + (chat - c) @ sae["dec_w"].T
            return fn

        def eval_transform(site, members_alpha, floor):
            sae = saes[site]
            ma = members_alpha.get(site, {})
            idx = torch.tensor(sorted(ma), device=DEV, dtype=torch.long)
            av = torch.tensor([ma[int(i)] for i in idx.tolist()],
                              device=DEV)
            fl = floor.get(site) if floor else None
            def fn(x):
                c = encode(sae, x)
                chat = (fl.expand_as(c).clone() if fl is not None
                        else torch.zeros_like(c))
                if len(idx):
                    chat[..., idx] = c[..., idx] * av
                return x + (chat - c) @ sae["dec_w"].T
            return fn

        def read(transforms, tokens, anchors):
            h = Hooks(transforms, ("resid", L), seed_sae, sl)
            pre = forward(tokens, h)
            B = pre.shape[0]
            v = pre[torch.arange(B, device=DEV),
                    anchors.to(DEV).clamp(0, pre.shape[1] - 1)]
            return float(F.relu(v).mean())

        e0_ho = read({s: eval_transform(s, {}, {}) for s in UP},
                     pos_ho, anchors_ho)
        eM_ho = read({s: eval_transform(s, {}, means_pos) for s in UP},
                     pos_ho, anchors_ho)
        print("[L%d %d] a_pos_ho %.3f | e0 %.3f eM %.3f | a_base %.3f"
              % (L, sl, a_pos_ho, e0_ho, eM_ho, a_base), flush=True)

        def fit(free_amp, lam, support=None):
            params = {}
            for site in UP:
                th0 = 40.0 if support is not None else 2.0
                th = torch.full((D_SAE,), -40.0 if support is not None
                                else th0, device=DEV)
                if support is not None and site in support:
                    th[support[site]] = 40.0
                th.requires_grad_(support is None)
                ps = torch.full((D_SAE,), math.log(math.e - 1.0),
                                device=DEV, requires_grad=free_amp)
                params[site] = (th, ps)
            opt_params = [p for pr in params.values() for p in pr
                          if p.requires_grad]
            opt = torch.optim.AdamW(opt_params, lr=LR, weight_decay=0.05)
            temp = [1.0]
            tnorm = float((target_tr ** 2).mean())
            floors3 = [(None, 1.0), (means_neg, DUAL_W),
                       (means_pos, TRIPLE_W)]
            for step in range(STEPS):
                temp[0] = 1.0 * (0.05 ** (step / max(STEPS - 1, 1)))
                s0 = (step * 4) % N_TRAIN
                tk = pos_tr[s0:s0 + 4]
                an = anchors_tr[s0:s0 + 4]
                tg = target_tr[s0:s0 + 4]
                opt.zero_grad()
                for fl, w in floors3:
                    tr = {s: spec_transform(s, None, temp, params,
                                            fl or {}) for s in UP}
                    pre = forward(tk, Hooks(tr, ("resid", L), seed_sae, sl),
                                  grad=True)
                    v = pre[torch.arange(pre.shape[0], device=DEV),
                            an.to(DEV)]
                    loss = w * ((v - tg.to(DEV)) ** 2).mean() / max(tnorm,
                                                                    1e-6)
                    loss.backward()
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
            members_alpha = {}
            with torch.no_grad():
                for site in UP:
                    th, ps = params[site]
                    m = torch.sigmoid(th / temp[0])
                    keep = (m > 0.5).nonzero(as_tuple=True)[0]
                    al = F.softplus(ps)[keep] if free_amp else \
                        torch.ones(len(keep), device=DEV)
                    members_alpha[site] = {int(i): float(a) for i, a in
                                           zip(keep.tolist(), al.tolist())}
            return members_alpha

        def score(ma, tag, secs):
            n = sum(len(d) for d in ma.values())
            aw0 = read({s: eval_transform(s, ma, {}) for s in UP},
                       pos_ho, anchors_ho)
            awM = read({s: eval_transform(s, ma, means_pos) for s in UP},
                       pos_ho, anchors_ho)
            # suppression: members zeroed, everything else LIVE
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
                    return x + (chat - c) @ _sae["dec_w"].T
                sup_tr[site] = fn
            a_sup = read(sup_tr, pos_ho, anchors_ho) if sup_tr else a_pos_ho
            # drive: members SET to alpha*pin in live negctx stream
            inj = {}
            for site in UP:
                d = ma.get(site, {})
                if not d:
                    continue
                sae = saes[site]
                idx = torch.tensor(sorted(d), device=DEV, dtype=torch.long)
                vals = torch.tensor(
                    [d[int(i)] * float(pins[site][int(i)])
                     for i in idx.tolist()], device=DEV)
                def fn(x, _sae=sae, _idx=idx, _vals=vals):
                    c = encode(_sae, x)
                    chat = c.clone()
                    chat[..., _idx] = _vals
                    return x + (chat - c) @ _sae["dec_w"].T
                inj[site] = fn
            a_inj = read(inj, neg_ho, na_ho) if inj else a_base
            row = {"layer": L, "latent": sl, "arm": tag, "n": n,
                   "ampF0": round((aw0 - e0_ho) / (a_pos_ho - e0_ho), 4)
                   if abs(a_pos_ho - e0_ho) > 1e-9 else None,
                   "ampFM": round((awM - eM_ho) / (a_pos_ho - eM_ho), 4)
                   if abs(a_pos_ho - eM_ho) > 1e-9 else None,
                   "sup": round(1.0 - a_sup / a_pos_ho, 4)
                   if a_pos_ho > 1e-9 else None,
                   "cf_amp": round((a_inj - a_base) / (a_pos_ho - a_base), 4)
                   if abs(a_pos_ho - a_base) > 1e-9 else None,
                   "a_pos_ho": round(a_pos_ho, 3),
                   "secs": round(secs, 1)}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("  %-9s n=%-6d ampF0=%-8s ampFM=%-8s sup=%-7s cf_amp=%s"
                  % (tag, n, row["ampF0"], row["ampFM"], row["sup"],
                     row["cf_amp"]), flush=True)
            return row

        n_ref = None
        if (L, sl, "triamp400" + TAG_SUFFIX) not in done:
            t0 = time.time()
            ma = fit(True, LAM)
            r = score(ma, "triamp400" + TAG_SUFFIX, time.time() - t0)
            n_ref = r["n"]
        if (L, sl, "gate400" + TAG_SUFFIX) not in done:
            t0 = time.time()
            mg = fit(False, LAM)
            score(mg, "gate400" + TAG_SUFFIX, time.time() - t0)
        if n_ref is None:
            for line in (HERE / "rows.jsonl").open():
                r = json.loads(line)
                if (r["layer"], r["latent"], r["arm"]) == (
                        L, sl, "triamp400" + TAG_SUFFIX):
                    n_ref = r["n"]
        # live pool for nulls: latents active anywhere on train posctx
        live = []
        for site in UP:
            cap = {}
            def grab(mod, inp, out, _s=site):
                x = out[0] if isinstance(out, tuple) else out
                cap[_s] = (encode(saes[_s], x) > 0).reshape(-1, D_SAE).any(0)
            hd = site_module(site).register_forward_hook(grab)
            with torch.no_grad():
                model(pos_tr.to(DEV), use_cache=False)
            hd.remove()
            live += [(site, int(i)) for i in
                     cap[site].nonzero(as_tuple=True)[0].tolist()]
        rng = random.Random(1000 + sl)
        for draw in range(N_NULL):
            members = rng.sample(live, min(n_ref, len(live)))
            tag = "null%d" % draw + TAG_SUFFIX
            if (L, sl, tag) in done:
                continue
            support = {}
            for s, i in members:
                support.setdefault(s, []).append(i)
            support = {s: torch.tensor(v, dtype=torch.long, device=DEV)
                       for s, v in support.items()}
            t0 = time.time()
            mn = fit(True, 0.0, support=support)
            score(mn, tag, time.time() - t0)
        del saes, seed_sae
        torch.cuda.empty_cache()
    fh.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    {"scan": scan, "run": run}[sys.argv[1]]()
