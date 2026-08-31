"""Our method (tri-amp mask) on Llama-3.2-1B + EleutherAI TopK skip
transcoders — the OURS side of the circuit-tracer comparison.

Both sides read the SAME converted weights (transcoders_llama_ct/), so
the feature space is shared and the comparison isolates the discovery
mechanism: linear attribution on a frozen-attention replacement model
(theirs) versus optimisation against the exact forward pass (ours).

Model        unsloth/Llama-3.2-1B (ungated mirror of meta-llama's)
Transcoders  EleutherAI/skip-transcoder-Llama-3.2-1B-131k
             TopK k=32, 131,072 latents, d_in 2048, skip connection,
             all 16 layers. Conventions were PROBED, not assumed
             (llama_loader.py): input = the MLP module's input, the
             source centres that input (folded into b_enc at convert
             time), and the skip term is essential (FVU 0.17 with it,
             0.94 without).

TRANSCODER INTERVENTION SEMANTICS
    c    = topk_k(x_in @ W_enc.T + b_enc)      [features]
    rec  = c @ W_dec + b_dec + x_in @ W_skip.T [predicted mlp_out]
    err  = mlp_out_true - rec                  [preserved]
An intervention c -> chat edits the MLP output by
    mlp_out <- mlp_out + (chat - c) @ W_dec
so an unmodified circuit reproduces the model exactly, and BOTH the
skip path and the transcoder's error pass through untouched. The skip
term is undecomposed computation that belongs to no feature — like
attention here — and it affects both methods identically.

Node universe: MLP-transcoder features at layers < seed layer.

  PYTHONPATH=. python ours_llama.py scan
  PYTHONPATH=. python ours_llama.py run
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
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
TC_DIR = HERE / "transcoders_llama_ct"
DEV = os.environ.get("DEV", "cuda")
DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[
    os.environ.get("DTYPE", "bf16")]
FIT_BS = int(os.environ.get("FIT_BS", 2))   # dense [B,T,131072] tensors
                                            # dominate memory here
D_MODEL, D_TC, N_LAYERS, K_SAE = 2048, 131072, 16, 32
SEQ_LEN, BATCH = 64, 32
N_POS, N_TRAIN = 64, 48
SEED_LAYERS = [int(x) for x in os.environ.get("SEED_LAYERS", "4,6").split(",")]
SEEDS_PER_LAYER = int(os.environ.get("SEEDS_PER_CELL", 3))
N_NULL = int(os.environ.get("N_NULL", 3))
# STEPS is overridable purely so a cheap smoke pass can exercise every
# arm and every scoring path (which is where the dtype seams live) in
# minutes rather than discovering them one crash at a time. Real runs
# use 400 and the arm tags below assume it.
STEPS = int(os.environ.get("STEPS", 400))
LR = 0.05
LAM = float(os.environ.get("LAM", 1e-3))


def _load_theirs(path):
    """circuit-tracer's size-matched node set per seed, if exported."""
    out = {}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("top_matched"):
                out[(r["layer"], r["latent"])] = r["top_matched"]
    return out


def _load_theirs_full(path):
    """Their ENTIRE ranking, not just the size-matched head.

    Scoring only their top-n would overclaim: a ranking that fails at n
    might still reconstruct given all the nodes it actually proposes.
    This arm separates "mis-ordered ranking" from "needs a bigger
    budget", which is the fair comparison to make.
    """
    out = {}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("ranking"):
                out[(r["layer"], r["latent"])] = [
                    [l, f] for l, f, _w in r["ranking"]]
    return out


def _load_ours_members(path):
    out = {}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            out[(r["layer"], r["latent"])] = r["alphas"]
    return out


# Scoring circuit-tracer's sets is opt-in so the original run is
# reproducible unchanged.
THEIRS_FILE = os.environ.get("THEIRS_FILE", "theirs_llama_nodes.jsonl")
ARM_PREFIX = os.environ.get("ARM_PREFIX", "theirs")
THEIRS_MULTS = [int(x) for x in os.environ.get("THEIRS_MULTS", "").split(",") if x]
HYB = os.environ.get("HYB") == "1"
LAMTAG = os.environ.get("LAMTAG", "")
THEIRS = (_load_theirs(THEIRS_FILE)
          if os.environ.get("SCORE_THEIRS") == "1" else {})
THEIRS_FULL = (_load_theirs_full(THEIRS_FILE)
               if os.environ.get("SCORE_THEIRS") == "1" else {})
def _load_ct_pruned(path):
    """circuit-tracer PRUNED circuits (as published / seed-pinned /
    seed-rooted), per seed: union sets and survival-frequency rankings,
    from the *_pruned.jsonl the driver writes."""
    out = {}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            d = {}
            for arm in ("ct_published", "ct_seed_pinned", "ct_seed_rooted"):
                a = r.get(arm)
                if a:
                    sz = sorted(a.get("size_per_window") or [])
                    d[arm] = {"union": [(int(l), int(f)) for l, f in a["union"]],
                              "freq": [(int(l), int(f)) for l, f, _ in a["freq"]],
                              "med": sz[len(sz) // 2] if sz else 0}
            out[(r["layer"], r["latent"])] = d
    return out


CT_PRUNED = (_load_ct_pruned("theirs_llama_pruned.jsonl")
             if os.environ.get("SCORE_CT") == "1" else {})
THEIRS_SELF_CHECK = os.environ.get("SELF_CHECK") == "1"
SUPPORT_NULL = os.environ.get("SUPPORT_NULL") == "1"
COACT = os.environ.get("COACT") == "1"
N_SUPNULL = int(os.environ.get("N_SUPNULL", 3))
OURS_MEMBERS = (_load_ours_members("ours_llama_members.jsonl")
                if THEIRS_SELF_CHECK or SUPPORT_NULL else {})
TRIPLE_W, DUAL_W = 0.10, 0.25
# HARD VRAM CAP, same guard as theirs_llama. On Windows/WDDM the driver
# silently oversubscribes into "shared GPU memory" (system RAM over
# PCIe, ~5x slower) rather than raising OOM, so a too-large config
# quietly crawls instead of failing. Capping the allocator makes it fail
# fast and loudly. Lower FIT_BS if this trips.
MEM_FRAC = float(os.environ.get("MEM_FRAC", 0.90))
if torch.cuda.is_available() and DEV == "cuda" and MEM_FRAC < 1.0:
    torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
torch.set_float32_matmul_precision("high")

MODEL_ID = "unsloth/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=DTYPE).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)

_TC = {}


def tc(layer):
    if layer not in _TC:
        sd = load_file(str(TC_DIR / ("layer_%d.safetensors" % layer)),
                       device=DEV)
        _TC[layer] = {k: v.to(DTYPE) for k, v in sd.items()}
    return _TC[layer]


def block(layer):
    return model.model.layers[layer]


def pre_acts(layer, x_in):
    """x_in is the MLP module's input (post_attention_layernorm output),
    the convention probed in llama_loader.py; the source's input centring
    is already folded into b_enc by the converter."""
    t = tc(layer)
    return x_in @ t["W_enc"].T + t["b_enc"]


def features(layer, x_in):
    """TopK code: keep the k largest pre-activations, zero the rest."""
    p = pre_acts(layer, x_in)
    vals, idx = p.topk(K_SAE, dim=-1)
    return torch.zeros_like(p).scatter(-1, idx, F.relu(vals))


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
            # `inp[0]` is the MLP module's input = transcoder input
            x_in = inp[0]
            c = features(layer, x_in)
            fn = self.transforms.get(layer)
            chat = fn(c) if fn is not None else c
            if fn is None:
                return None
            # skip path and reconstruction error are untouched.
            # The mask/amplitude parameters are fp32 (so the optimiser and
            # the loss keep full precision), while the stream is bf16, so
            # the DELTA is cast at the matmul boundary rather than the
            # parameters being downcast. Autograd carries the gradient
            # back through the cast, so this costs precision only in the
            # injected residual, not in what is being fitted.
            return out + (chat - c).to(out.dtype) @ tc(layer)["W_dec"]
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
    """Token windows, each PREPENDED WITH BOS.

    circuit-tracer's ensure_tokenized() prepends a special token before
    attribution (position 0 carries anomalous norm and feature counts,
    so they add one and ignore it). Our windows must match exactly, or
    the two sides would build their circuits on different sequences and
    at shifted positions."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="Salesforce/wikitext", repo_type="dataset",
        filename="wikitext-103-raw-v1/train-00000-of-00002.parquet")
    texts = pq.read_table(path, columns=["text"])["text"].to_pylist()
    bos = tok.bos_token_id
    assert bos is not None, "tokenizer has no BOS to match ensure_tokenized"
    content = SEQ_LEN - 1
    buf, out = [], []
    for t_ in texts:
        if not t_.strip():
            continue
        buf.extend(tok(t_, add_special_tokens=False)["input_ids"])
        while len(buf) >= content:
            out.append([bos] + buf[:content]); buf = buf[content:]
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
    torch.save({"tokens": toks, "seeds": stats}, HERE / "scan_llama.pt")
    print("SCAN DONE (%d seeds)" % len(stats), flush=True)


def run():
    data = torch.load(HERE / "scan_llama.pt", weights_only=False)
    toks, seeds = data["tokens"], data["seeds"]
    rows_path = HERE / "ours_llama_rows.jsonl"
    mem_path = HERE / "ours_llama_members.jsonl"
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
        # "Complete" has to account for the opt-in arms too, or a seed
        # finished before they existed skips before ever reaching them.
        needed = {"null%d" % (N_NULL - 1)}
        if LAMTAG:
            needed |= {"triamp400" + LAMTAG, "gate400" + LAMTAG}
        if HYB and THEIRS.get((L, sl)):
            needed.add(ARM_PREFIX + "_amp")
        if THEIRS_MULTS and THEIRS_FULL.get((L, sl)):
            # n_ref is not known yet here; over-include the mult tags.
            # Invalid cuts are skipped at scoring time, so the only cost
            # of over-inclusion is re-entering an already-complete seed.
            for _m in THEIRS_MULTS:
                needed.add("%s_x%d" % (ARM_PREFIX, _m))
        if HYB and CT_PRUNED.get((L, sl), {}).get("ct_seed_rooted"):
            needed.add("ct_seed_rooted_matched_amp")
        if THEIRS.get((L, sl)):
            needed.add(ARM_PREFIX)
        if THEIRS_FULL.get((L, sl)):
            needed.add(ARM_PREFIX + "_full")
        if THEIRS_SELF_CHECK and OURS_MEMBERS.get((L, sl)):
            needed.add("selfcheck")
        if SUPPORT_NULL and OURS_MEMBERS.get((L, sl)):
            needed.add("nullsup%d" % (N_SUPNULL - 1))
        if COACT:
            needed.add("coact_amp")
        for _arm, _d in (CT_PRUNED.get((L, sl)) or {}).items():
            if _d["union"]:
                needed.add("%s_union" % _arm)
        if all((L, sl, a) in done for a in needed):
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

        # circuit-tracer ignores position 0 (prepended BOS); exclude it
        # from anchor selection so both methods read the same positions.
        nat_tr = natural(pos_tr)
        nat_tr[:, 0] = -float("inf")
        anchors_tr = nat_tr.argmax(dim=1)
        target_tr = nat_tr.gather(1, anchors_tr[:, None]).squeeze(1)
        nat_ho = natural(pos_ho)
        nat_ho[:, 0] = -float("inf")
        anchors_ho = nat_ho.argmax(dim=1)
        a_pos_ho = float(F.relu(nat_ho.gather(1, anchors_ho[:, None])).mean())
        neg_pre = natural(neg_ho)
        neg_pre[:, 0] = -float("inf")
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
                # fitted amplitudes arrive as Python floats; they are
                # written into a code tensor that follows the stream dtype
                av = torch.tensor([d[int(i)] for i in idx.tolist()],
                                  device=DEV, dtype=DTYPE)
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
            tnorm = max(float((target_tr.float() ** 2).mean()), 1e-6)
            floors3 = [(None, 1.0), (means_neg, DUAL_W), (means_pos, TRIPLE_W)]
            for step in range(STEPS):
                temp[0] = 1.0 * (0.05 ** (step / max(STEPS - 1, 1)))
                s0 = (step * FIT_BS) % N_TRAIN
                tk = pos_tr[s0:s0 + FIT_BS]
                an = anchors_tr[s0:s0 + FIT_BS]
                tg = target_tr[s0:s0 + FIT_BS]
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
                    v = pre[torch.arange(pre.shape[0], device=DEV),
                            an.to(DEV)].float()
                    (w * ((v - tg.to(DEV).float()) ** 2).mean()
                     / tnorm).backward()
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
                                     for i in idx.tolist()], device=DEV,
                                    dtype=DTYPE)
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
        if (L, sl, "triamp400" + LAMTAG) not in done:
            t0 = time.time()
            n_ref = score(fit(True, LAM), "triamp400" + LAMTAG, time.time() - t0)["n"]
        if (L, sl, "gate400" + LAMTAG) not in done:
            t0 = time.time()
            score(fit(False, LAM), "gate400" + LAMTAG, time.time() - t0)
        if n_ref is None:
            for line in rows_path.open():
                r = json.loads(line)
                if (r["layer"], r["latent"], r["arm"]) == (L, sl,
                                                           "triamp400"):
                    n_ref = r["n"]

        # circuit-tracer's node set, scored through the SAME score()
        # closure as our own arms -- deliberately not a parallel
        # implementation, so any difference is a property of the SETS and
        # not of two scoring stacks. Their method outputs a ranking with
        # no per-latent coefficients, so it is rendered at alpha = 1.0:
        # a membership-only circuit, directly comparable to gate400.
        # THEIRS_SELF_CHECK re-scores OUR OWN members through this path
        # and must reproduce the triamp400 row, which is what certifies
        # the two are really the same code path.
        if THEIRS and (L, sl, ARM_PREFIX) not in done:
            tm = THEIRS.get((L, sl))
            if tm:
                ma = {}
                for lyr, f in tm:
                    ma.setdefault(int(lyr), {})[int(f)] = 1.0
                score(ma, ARM_PREFIX, 0.0)
        if HYB and THEIRS and (L, sl, ARM_PREFIX + "_amp") not in done:
            tm = THEIRS.get((L, sl))
            if tm:
                support = {}
                for lyr, f in tm:
                    support.setdefault(int(lyr), []).append(int(f))
                support = {k: torch.tensor(v, dtype=torch.long, device=DEV)
                           for k, v in support.items()}
                t0 = time.time()
                score(fit(True, 0.0, support=support), ARM_PREFIX + "_amp",
                      time.time() - t0)
        # size multiples of the external ranking, CAPPED at its length:
        # short rankings (a property of the substrate, e.g. 128-758 direct
        # edges on TopK) contribute points at their true size rather than
        # being skipped, and duplicates of matched/full are suppressed.
        if THEIRS_FULL and THEIRS_MULTS and n_ref:
            tf = THEIRS_FULL.get((L, sl)) or []
            for m in THEIRS_MULTS:
                cut = min(n_ref * m, len(tf))
                tag = "%s_x%d" % (ARM_PREFIX, m)
                if (L, sl, tag) in done or cut <= n_ref or cut >= len(tf):
                    continue
                ma = {}
                for lyr, f in tf[:cut]:
                    ma.setdefault(int(lyr), {})[int(f)] = 1.0
                score(ma, tag, 0.0)
        if THEIRS_FULL and (L, sl, ARM_PREFIX + "_full") not in done:
            tf = THEIRS_FULL.get((L, sl))
            if tf:
                ma = {}
                for lyr, f in tf:
                    ma.setdefault(int(lyr), {})[int(f)] = 1.0
                score(ma, ARM_PREFIX + "_full", 0.0)

        # circuit-tracer PRUNED objects, alpha=1 (their method emits no
        # coefficients). Each contributes a POINT at its natural size
        # (union across windows; per-window-median-sized cut of the
        # survival-frequency ranking) and a matched-size cut for the
        # paired comparison. Same score() closure as every other arm.
        for _arm, _d in (CT_PRUNED.get((L, sl)) or {}).items():
            _cuts = [("%s_union" % _arm, _d["union"]),
                     ("%s_med" % _arm, _d["freq"][:_d["med"]]),
                     ("%s_matched" % _arm,
                      _d["freq"][:n_ref] if n_ref else [])]
            for _tag, _mem in _cuts:
                if (L, sl, _tag) in done or not _mem:
                    continue
                _ma = {}
                for _lyr, _f in _mem:
                    _ma.setdefault(int(_lyr), {})[int(_f)] = 1.0
                score(_ma, _tag, 0.0)
        if HYB and CT_PRUNED.get((L, sl), {}).get("ct_seed_rooted") and n_ref \
                and (L, sl, "ct_seed_rooted_matched_amp") not in done:
            _mem = CT_PRUNED[(L, sl)]["ct_seed_rooted"]["freq"][:n_ref]
            if _mem:
                support = {}
                for _lyr, _f in _mem:
                    support.setdefault(int(_lyr), []).append(int(_f))
                support = {k: torch.tensor(v, dtype=torch.long, device=DEV)
                           for k, v in support.items()}
                t0 = time.time()
                score(fit(True, 0.0, support=support),
                      "ct_seed_rooted_matched_amp", time.time() - t0)
        if THEIRS_SELF_CHECK and (L, sl, "selfcheck") not in done:
            om = OURS_MEMBERS.get((L, sl))
            if om:
                score({int(k): {int(i): float(a) for i, a in v.items()}
                       for k, v in om.items()}, "selfcheck", 0.0)
        live, anchor_rate, acount, aval = [], [], {}, {}
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
            acount[layer] = at.sum(0)     # per-latent anchor firings
            aval[layer] = f[torch.arange(f.shape[0], device=DEV),
                            anchors_tr.to(DEV)].sum(0)  # anchor mass
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

        # SUPPORT-MATCHED NULL.
        #
        # The plain null above draws from every latent that fires
        # ANYWHERE. Under TopK only ~4% of those are active AT THE
        # ANCHOR, so most drawn latents are exactly zero there and no
        # fitted amplitude can rescue them (alpha * 0 = 0). That makes
        # the plain null easy for a reason that is a property of the
        # DICTIONARY, not of our selection -- the mirror image of why it
        # was vacuous on ReLU/L1 banks, where a third of the pool is live
        # at the anchor and a random set reconstructs on its own.
        #
        # This null instead draws only from latents that fire at the
        # anchor, matched to each member's anchor-firing COUNT. It
        # therefore controls for support, and means the same thing on any
        # architecture. If our circuits still beat it, the result is
        # about SELECTION rather than about how sparse the bank happens
        # to be.
        if SUPPORT_NULL and n_ref and OURS_MEMBERS.get((L, sl)):
            mem = OURS_MEMBERS[(L, sl)]
            members = [(int(k), int(i)) for k, v in mem.items() for i in v]
            mset = set(members)
            cand = {}
            for lyr in UP:
                cnt = acount[lyr]
                for i in (cnt > 0).nonzero(as_tuple=True)[0].tolist():
                    if (lyr, i) not in mset:
                        cand.setdefault(int(cnt[i]), []).append((lyr, i))
            counts = sorted(cand)
            m_counts = [int(acount[l][i]) for l, i in members]
            for draw in range(N_SUPNULL):
                tag = "nullsup%d" % draw
                if (L, sl, tag) in done or not counts:
                    continue
                rng2 = random.Random(5000 + sl * 10 + draw)
                for v in cand.values():
                    rng2.shuffle(v)
                used, picked, pc = {}, [], []
                for c in m_counts:
                    best = None
                    for cc in sorted(counts, key=lambda x: (abs(x - c), x)):
                        if used.get(cc, 0) < len(cand[cc]):
                            best = cc
                            break
                    if best is None:
                        break
                    picked.append(cand[best][used.get(best, 0)])
                    used[best] = used.get(best, 0) + 1
                    pc.append(best)
                if not picked:
                    continue
                support = {}
                for lyr, i in picked:
                    support.setdefault(lyr, []).append(i)
                support = {k: torch.tensor(v, dtype=torch.long, device=DEV)
                           for k, v in support.items()}
                t0 = time.time()
                r = score(fit(True, 0.0, support=support), tag,
                          time.time() - t0)
                # report the match quality alongside, so a null that
                # failed to match on support is not read as a real null
                print("      support match: ours mean %.2f firings/anchor "
                      "vs null %.2f (n %d)"
                      % (sum(m_counts) / max(len(m_counts), 1),
                         sum(pc) / max(len(pc), 1), len(picked)), flush=True)
        # RUNG 3 (see 037-gemmascope): greedy co-activation
        # baseline, the obvious simple discovery method. Top-n latents by
        # summed anchor activation, size-matched to tri-amp; scored raw
        # (alpha=1) and amplitude-fitted through the same score() path.
        if COACT and n_ref:
            allv = torch.cat([aval[l] for l in UP])
            top = allv.argsort(descending=True)[:n_ref]
            members = [(UP[int(i) // D_TC], int(i) % D_TC)
                       for i in top.tolist()]
            if (L, sl, "coact_raw") not in done:
                ma = {}
                for lyr, i in members:
                    ma.setdefault(lyr, {})[i] = 1.0
                score(ma, "coact_raw", 0.0)
            if (L, sl, "coact_amp") not in done:
                support = {}
                for lyr, i in members:
                    support.setdefault(lyr, []).append(i)
                support = {k: torch.tensor(v, dtype=torch.long, device=DEV)
                           for k, v in support.items()}
                t0 = time.time()
                score(fit(True, 0.0, support=support), "coact_amp",
                      time.time() - t0)
        torch.cuda.empty_cache()
    fh.close(); mh.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    {"scan": scan, "run": run}[sys.argv[1]]()
