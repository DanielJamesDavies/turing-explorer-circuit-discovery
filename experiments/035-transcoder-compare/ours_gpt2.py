"""Our method (tri-amp mask) on GPT-2 small + Dunefsky/Chlenski
per-layer transcoders — the OURS side of the circuit-tracer comparison.

Both sides read the SAME converted weights (transcoders_ct/), so the
feature space is shared and the comparison isolates the discovery
mechanism: linear attribution on a frozen-attention replacement model
(theirs) versus optimisation against the exact forward pass (ours).

TRANSCODER INTERVENTION SEMANTICS (the one new piece of machinery).
A per-layer transcoder reads the MLP input (ln2-normalised residual)
and predicts the MLP output:
    c   = relu(x_in @ W_enc.T + b_enc)         [features]
    rec = c @ W_dec + b_dec                    [predicted mlp_out]
    err = mlp_out_true - rec                   [preserved, as with SAEs]
An intervention on features c -> chat therefore edits the MLP output by
    mlp_out <- mlp_out + (chat - c) @ W_dec
so an unmodified circuit reproduces the model exactly and the
transcoder's own error passes through untouched — the same delta
algebra as our SAE harnesses, one level of indirection out.

Node universe: MLP-transcoder features at layers < seed layer. These
transcoders decompose MLP outputs only, so attention and embeddings
stay live and undecomposed — which is also true of circuit-tracer's
PLT graphs, so both methods work in the same universe.

Seed read: the transcoder feature's PRE-activation (before ReLU), the
same quantity our SAE work uses.

  PYTHONPATH=. python ours_gpt2.py scan
  PYTHONPATH=. python ours_gpt2.py run
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
from safetensors.torch import load_file
from transformers import AutoTokenizer, GPT2LMHeadModel

HERE = Path(__file__).parent
TC_DIR = HERE / "transcoders_ct"
DEV = os.environ.get("DEV", "cuda")
D_MODEL, D_TC, N_LAYERS = 768, 24576, 12
SEQ_LEN, BATCH = 64, 32
N_POS, N_TRAIN = 64, 48
SEED_LAYERS = [int(x) for x in os.environ.get("SEED_LAYERS", "4,6,8").split(",")]
SEEDS_PER_LAYER = int(os.environ.get("SEEDS_PER_CELL", 3))
N_NULL = int(os.environ.get("N_NULL", 3))
STEPS, LR = 400, 0.05
LAM = float(os.environ.get("LAM", 1e-3))
TRIPLE_W, DUAL_W = 0.10, 0.25
torch.set_float32_matmul_precision("high")

tok = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").float().to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)

_TC = {}


def tc(layer):
    if layer not in _TC:
        sd = load_file(str(TC_DIR / ("layer_%d.safetensors" % layer)),
                       device=DEV)
        _TC[layer] = {k: v.float() for k, v in sd.items()}
    return _TC[layer]


def block(layer):
    return model.transformer.h[layer]


def mlp_input(layer, hidden):
    """The transcoder's input: the FULL ln_2 output (weight and bias
    applied). The converted weights fold GPT-2's LayerNorm affine into
    the encoder for exactly this reason — see convert_transcoders.py."""
    return block(layer).ln_2(hidden)


def pre_acts(layer, x_in):
    t = tc(layer)
    return x_in @ t["W_enc"].T + t["b_enc"]


def features(layer, x_in):
    return F.relu(pre_acts(layer, x_in))


def recon(layer, feats):
    t = tc(layer)
    return feats @ t["W_dec"] + t["b_dec"]


class Runner:
    """Runs GPT-2 with per-layer transcoder-feature interventions on the
    MLP outputs, capturing the seed feature's pre-activation."""

    def __init__(self, transforms, seed_layer=None, seed_idx=None):
        self.transforms = transforms          # {layer: fn(feats)->feats}
        self.seed_layer, self.seed_idx = seed_layer, seed_idx
        self.seed_out = None
        self.handles = []

    def _mlp_hook(self, layer):
        def hook(mod, inp, out):
            # `inp[0]` is the ln_2-normalised residual = transcoder input
            x_in = inp[0]
            c = features(layer, x_in)
            fn = self.transforms.get(layer)
            chat = fn(c) if fn is not None else c
            if fn is None:
                return None
            return out + (chat - c) @ tc(layer)["W_dec"]
        return hook

    def _seed_hook(self):
        def hook(mod, inp, out):
            self.seed_out = pre_acts(self.seed_layer, inp[0])[
                ..., self.seed_idx]
            return None
        return hook

    def __enter__(self):
        for layer in self.transforms:
            self.handles.append(
                block(layer).mlp.register_forward_hook(self._mlp_hook(layer)))
        if self.seed_layer is not None:
            self.handles.append(
                block(self.seed_layer).mlp.register_forward_hook(
                    self._seed_hook()))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def forward(tokens, runner, grad=False):
    with torch.set_grad_enabled(grad), runner:
        model(tokens.to(DEV))
    return runner.seed_out


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
        buf.extend(tok(t_)["input_ids"])
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
        fire = torch.zeros(D_TC, device=DEV)
        mx = torch.zeros(D_TC, device=DEV)
        chunks = []
        with torch.no_grad():
            for s0 in range(0, len(toks), BATCH):
                cap = {}
                hd = block(L).mlp.register_forward_hook(
                    lambda m, i, o: cap.__setitem__("f", features(L, i[0])))
                model(toks[s0:s0 + BATCH].to(DEV))
                hd.remove()
                wmax = cap["f"].amax(dim=1)
                fire += (wmax > 0).float().sum(0)
                mx = torch.maximum(mx, wmax.amax(0))
                chunks.append(wmax.half().cpu())
        frac = fire / len(toks)
        band = ((frac > 0.005) & (frac < 0.05)).nonzero(as_tuple=True)[0]
        if not len(band):
            print("L%d: no seeds in band" % L, flush=True)
            continue
        chosen = band[mx[band].argsort(descending=True)[:SEEDS_PER_LAYER]]
        wmax_all = torch.cat(chunks)
        for sl in chosen.tolist():
            col = wmax_all[:, sl]
            top = col.argsort(descending=True)[:N_POS].tolist()
            silent = (col == 0).nonzero(as_tuple=True)[0]
            neg = silent[torch.randperm(len(silent))[:N_POS]].tolist()
            stats["%d/%d" % (L, sl)] = {
                "layer": L, "latent": int(sl),
                "fire_frac": round(float(frac[sl]), 4),
                "max_act": round(float(mx[sl]), 3),
                "pos_windows": top, "neg_windows": neg}
            print("L%d feature %d: frac %.3f max %.2f"
                  % (L, sl, frac[sl], mx[sl]), flush=True)
        del wmax_all, chunks
    torch.save({"tokens": toks, "seeds": stats}, HERE / "scan.pt")
    print("SCAN DONE (%d seeds)" % len(stats), flush=True)


def run():
    data = torch.load(HERE / "scan.pt", weights_only=False)
    toks, seeds = data["tokens"], data["seeds"]
    rows_path = HERE / "ours_rows.jsonl"
    mem_path = HERE / "ours_members.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.open():
            try:
                r = json.loads(line)
                done.add((r["layer"], r["latent"], r["arm"]))
            except Exception:
                pass
    fh, mh = rows_path.open("a"), mem_path.open("a")

    for key, S in seeds.items():
        L, sl = S["layer"], S["latent"]
        if (L, sl, "null%d" % (N_NULL - 1)) in done:
            print("[L%d %d] complete, skipping" % (L, sl), flush=True)
            continue
        UP = list(range(L))            # MLP transcoder layers below the seed
        if not UP:
            continue
        pos, neg = toks[S["pos_windows"]], toks[S["neg_windows"]]
        pos_tr, pos_ho = pos[:N_TRAIN], pos[N_TRAIN:]
        neg_tr, neg_ho = neg[:N_TRAIN], neg[N_TRAIN:]

        def natural(t):
            return forward(t, Runner({}, L, sl)).detach()

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
            print("[L%d %d] held-out a_pos too small, skip" % (L, sl),
                  flush=True)
            continue

        def feat_means(t):
            means = {}
            for layer in UP:
                cap = {}
                hd = block(layer).mlp.register_forward_hook(
                    lambda m, i, o, _l=layer: cap.__setitem__(
                        _l, features(_l, i[0]).mean(dim=(0, 1))))
                with torch.no_grad():
                    model(t.to(DEV))
                hd.remove()
                means[layer] = cap[layer].detach()
            return means

        means_pos, means_neg = feat_means(pos_tr), feat_means(neg_tr)
        pins = {}
        for layer in UP:
            cap = {}
            hd = block(layer).mlp.register_forward_hook(
                lambda m, i, o, _l=layer: cap.__setitem__(
                    _l, features(_l, i[0])))
            with torch.no_grad():
                model(pos_tr.to(DEV))
            hd.remove()
            f = cap[layer]
            pins[layer] = f[torch.arange(f.shape[0], device=DEV),
                            anchors_tr.to(DEV)].mean(0).detach()

        def eval_tf(ma, floor):
            out = {}
            for layer in UP:
                d = ma.get(layer, {})
                idx = torch.tensor(sorted(d), device=DEV, dtype=torch.long)
                av = torch.tensor([d[int(i)] for i in idx.tolist()],
                                  device=DEV)
                fl = floor.get(layer) if floor else None
                def fn(c, _idx=idx, _av=av, _fl=fl):
                    chat = (_fl.expand_as(c).clone() if _fl is not None
                            else torch.zeros_like(c))
                    if len(_idx):
                        chat[..., _idx] = c[..., _idx] * _av
                    return chat
                out[layer] = fn
            return out

        def read(transforms, tokens, anchors):
            pre = forward(tokens, Runner(transforms, L, sl))
            B = pre.shape[0]
            return float(F.relu(pre[torch.arange(B, device=DEV),
                                    anchors.to(DEV)]).mean())

        e0 = read(eval_tf({}, {}), pos_ho, anchors_ho)
        eM = read(eval_tf({}, means_pos), pos_ho, anchors_ho)
        print("\n[L%d %d] a_pos_ho %.3f | e0 %.3f eM %.3f | a_base %.3f"
              % (L, sl, a_pos_ho, e0, eM, a_base), flush=True)

        def fit(free_amp, lam, support=None):
            params = {}
            for layer in UP:
                th = torch.full((D_TC,), -40.0 if support is not None
                                else 2.0, device=DEV)
                if support is not None and layer in support:
                    th[support[layer]] = 40.0
                th.requires_grad_(support is None)
                ps = torch.full((D_TC,), math.log(math.e - 1.0), device=DEV,
                                requires_grad=free_amp)
                params[layer] = (th, ps)
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
                    tr = {}
                    for layer in UP:
                        def fn(c, _l=layer, _fl=fl):
                            th, ps = params[_l]
                            m = torch.sigmoid(th / temp[0])
                            chat = m * F.softplus(ps) * c
                            if _fl is not None:
                                chat = chat + (1 - m) * _fl[_l]
                            return chat
                        tr[layer] = fn
                    pre = forward(tk, Runner(tr, L, sl), grad=True)
                    v = pre[torch.arange(pre.shape[0], device=DEV), an.to(DEV)]
                    (w * ((v - tg.to(DEV)) ** 2).mean() / tnorm).backward()
                pen = 0.0
                for layer in UP:
                    th, ps = params[layer]
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
                for layer in UP:
                    th, ps = params[layer]
                    keep = (torch.sigmoid(th / temp[0]) > 0.5).nonzero(
                        as_tuple=True)[0]
                    al = (F.softplus(ps)[keep] if free_amp
                          else torch.ones(len(keep), device=DEV))
                    out[layer] = {int(i): float(a) for i, a in
                                  zip(keep.tolist(), al.tolist())}
            return out

        def score(ma, tag, secs):
            n = sum(len(d) for d in ma.values())
            aw0 = read(eval_tf(ma, {}), pos_ho, anchors_ho)
            awM = read(eval_tf(ma, means_pos), pos_ho, anchors_ho)
            sup_tf = {}
            for layer in UP:
                d = ma.get(layer, {})
                if not d:
                    continue
                idx = torch.tensor(sorted(d), device=DEV, dtype=torch.long)
                def fn(c, _idx=idx):
                    chat = c.clone()
                    chat[..., _idx] = 0.0
                    return chat
                sup_tf[layer] = fn
            a_sup = read(sup_tf, pos_ho, anchors_ho) if sup_tf else a_pos_ho
            inj = {}
            for layer in UP:
                d = ma.get(layer, {})
                if not d:
                    continue
                idx = torch.tensor(sorted(d), device=DEV, dtype=torch.long)
                vals = torch.tensor([d[int(i)] * float(pins[layer][int(i)])
                                     for i in idx.tolist()], device=DEV)
                def fn(c, _idx=idx, _v=vals):
                    chat = c.clone()
                    chat[..., _idx] = _v
                    return chat
                inj[layer] = fn
            a_inj = read(inj, neg_ho, na_ho) if inj else a_base
            row = {"layer": L, "latent": sl, "arm": tag, "n": n,
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
            if tag == "triamp400":
                mh.write(json.dumps({
                    "layer": L, "latent": sl, "arm": tag,
                    "members": {str(k): sorted(v) for k, v in ma.items()},
                    "alphas": {str(k): {str(i): round(a, 4)
                                        for i, a in v.items()}
                               for k, v in ma.items()}}) + "\n")
                mh.flush()
            print("  %-11s n=%-6d ampF0=%-8s ampFM=%-8s sup=%-7s cf=%s"
                  % (tag, n, row["ampF0"], row["ampFM"], row["sup"],
                     row["cf_amp"]), flush=True)
            return row

        n_ref = None
        if (L, sl, "triamp400") not in done:
            t0 = time.time()
            n_ref = score(fit(True, LAM), "triamp400", time.time() - t0)["n"]
        if (L, sl, "gate400") not in done:
            t0 = time.time()
            score(fit(False, LAM), "gate400", time.time() - t0)
        if n_ref is None:
            for line in rows_path.open():
                r = json.loads(line)
                if (r["layer"], r["latent"], r["arm"]) == (L, sl,
                                                           "triamp400"):
                    n_ref = r["n"]
        live, anchor_rate = [], []
        for layer in UP:
            cap = {}
            hd = block(layer).mlp.register_forward_hook(
                lambda m, i, o, _l=layer: cap.__setitem__(
                    _l, features(_l, i[0])))
            with torch.no_grad():
                model(pos_tr.to(DEV))
            hd.remove()
            f = cap[layer]
            lm = (f > 0).reshape(-1, D_TC).any(0)
            live += [(layer, int(i)) for i in lm.nonzero(as_tuple=True)[0]
                     .tolist()]
            at = (f[torch.arange(f.shape[0], device=DEV),
                    anchors_tr.to(DEV)] > 0).float()
            if lm.any():
                anchor_rate.append(float(at[:, lm].mean()))
        rate = sum(anchor_rate) / max(len(anchor_rate), 1)
        fh.write(json.dumps({"layer": L, "latent": sl,
                             "arm": "anchor_support", "n": len(live),
                             "anchor_support_rate": round(rate, 5)}) + "\n")
        fh.flush()
        print("  anchor support %.5f over live pool %d" % (rate, len(live)),
              flush=True)
        if n_ref:
            rng = random.Random(1000 + sl)
            for draw in range(N_NULL):
                members = rng.sample(live, min(n_ref, len(live)))
                tag = "null%d" % draw
                if (L, sl, tag) in done:
                    continue
                support = {}
                for lyr, i in members:
                    support.setdefault(lyr, []).append(i)
                support = {k: torch.tensor(v, dtype=torch.long, device=DEV)
                           for k, v in support.items()}
                t0 = time.time()
                score(fit(True, 0.0, support=support), tag, time.time() - t0)
        torch.cuda.empty_cache()
    fh.close(); mh.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    {"scan": scan, "run": run}[sys.argv[1]]()
