"""GREATER-THAN AS A COMPUTATION OVER ITS MEMBERS: frame x tens -> threshold.

"The {noun} lasted from the year 19{T}{U} to the year 19" -> digit > T.
The task-selective set (task_circuit_sets.json['gt'], 15 circuits, ~1.9k
member latents) is read as a computation:

  1. ROLE-ASSIGN EVERY MEMBER from its activations on the 255 prompts:
       FRAME     fires on >= 90% of prompts at the century "9" (c2), the
                 first "1"/"9" of the final year (f1/f2), flat in T
       TENS      at the tens digit: a DETECTOR (one digit >= 70%, others
                 <= 20%) or GRADED (|corr(act, T)| >= 0.6)
       TRANSPORT at the final position f2: activation depends on T
                 (graded / detector) although T is 6 tokens back
       OUTPUT    at f2: direct-logit effect over the ten digits that is
                 monotone in the digit (promotes high / suppresses low)
  2. THE TENS SWAP: for receiver prompts with tens T_r and donor prompts
     with tens T_d (|T_d - T_r| >= 3, same noun where possible), install
     the donor's TENS-reader states at the tens position (and separately
     the donor's TRANSPORT states at the final position). If the members
     carry the variable, the model's threshold moves to T_d:
        gap mass G = P(digit in (min(T_r,T_d), max(T_r,T_d)])
        swap score = (G_swap - G_receiver) / (G_donor - G_receiver)
     1 = behaves exactly like the donor tens, 0 = unchanged. Controls:
     size/site-matched random latents given the same donor values; the
     whole set at those positions as the upper bound.

  PYTHONPATH=src python experiments/049-circuit-graph/gt_variables.py
Env: N_PAIRS (60), MIN_GAP (3)
"""
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from model.tokenizer import Tokenizer
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense

HERE = Path(__file__).parent
T = Path(os.environ.get("TABLES", str(HERE / "tables_full")))
R = Path(os.environ.get("OUT", str(HERE / "results_full")))
D047 = HERE.parent / "047-known-circuits"
N_PAIRS = int(os.environ.get("N_PAIRS", 60)); MIN_GAP = int(os.environ.get("MIN_GAP", 3))
EVAL_BS = 16
rng = np.random.default_rng(0)
torch.set_float32_matmul_precision("high")
devices = detect_devices(); device = devices[0]
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
tok = Tokenizer(); KINDS = list(bank.kinds); D = bank.d_sae
inference.disable_compile()

C = pd.read_parquet(T / "circuits.parquet")
M = pd.read_parquet(T / "members.parquet"); M["kind"] = M["kind"].astype(str)
seed_of = {int(r.cid): (int(r.seed_layer), str(r.seed_kind), int(r.seed_index)) for r in C.itertuples()}
skey = {c: "%d.%s.%d" % s for c, s in seed_of.items()}
cid_of = {v: k for k, v in skey.items()}
members = defaultdict(lambda: defaultdict(set)); amps = defaultdict(lambda: defaultdict(dict))
for cid, l, k, i, a in zip(M["cid"].values, M["layer"].values, M["kind"].values, M["index"].values, M["amplitude"].values):
    members[int(cid)][(int(l), k)].add(int(i)); amps[int(cid)][(int(l), k)][int(i)] = float(a) if a == a else 1.0
fo = pd.read_csv(R / "fanout_latents.csv")
HUBS = defaultdict(set)
for _, r in fo[fo["fanout"] >= 0.05 * len(C)].iterrows():
    HUBS[(int(r["layer"]), str(r["kind"]))].add(int(r["index"]))

# ---- data (as gt_mechanism.py) ------------------------------------------------------------
d = torch.load(D047 / "gt_clusters.pt", weights_only=False)
digit_ids = [int(x) for x in d["digit_ids"]]
rows = []
for i, w in enumerate(d["windows"]):
    a = int(d["anchors"][i]); ids = [int(x) for x in w[:a + 1]]
    tens = int(d["tens"][i]); target = int(d["targets"][i])
    words = [tok.decode([t]) for t in ids]
    dpos = [p for p, t in enumerate(ids) if t in digit_ids]
    if len(dpos) < 6:
        continue
    roles = ["other"] * len(ids)
    roles[dpos[0]] = "c1"; roles[dpos[1]] = "c2"; roles[dpos[2]] = "tens"; roles[dpos[3]] = "units"
    roles[dpos[-2]] = "f1"; roles[dpos[-1]] = "f2"
    prompt = d["prompts"][i]
    rows.append(dict(ids=ids, roles=roles, tens=tens, target=target, prompt=prompt,
                     noun=prompt.split(" lasted")[0] if " lasted" in prompt else prompt[:20],
                     rpos={ro: p for p, ro in enumerate(roles) if ro != "other"}))
tens_of = np.array([r["tens"] for r in rows])
print("gt prompts: %d | tens histogram %s | nouns %d" % (len(rows), dict(sorted(Counter(tens_of).items())), len({r["noun"] for r in rows})))

sel = json.load(open(R / "task_circuit_sets.json"))["gt"]["circuits"]
cids = [cid_of[s] for s in sel]
U = defaultdict(set)                      # the member universe, hubs excluded
for c in cids:
    l, k, i = seed_of[c]; U[(l, k)].add(i)
    for s, v in members[c].items():
        U[s] |= v
U = {s: v - HUBS.get(s, set()) for s, v in U.items()}
AMP = defaultdict(dict)                   # median amplitude across the circuits that contain the latent
tmp = defaultdict(list)
for c in cids:
    for s, dd in amps[c].items():
        for i, a in dd.items():
            tmp[(s, i)].append(a)
for (s, i), v in tmp.items():
    AMP[s][i] = float(np.median(v))
n_u = sum(len(v) for v in U.values())
print("member universe: %d latents (hubs excluded) over %d sites" % (n_u, len(U)))


def batches(rs):
    for s in range(0, len(rs), EVAL_BS):
        b = rs[s:s + EVAL_BS]
        L = max(len(r["ids"]) for r in b)
        tk = torch.tensor([r["ids"] + [0] * (L - len(r["ids"])) for r in b], dtype=torch.long, device=device)
        yield b, tk


class CaptureU:
    """dense activations of the universe latents at every position: {site: [B, T, n]}."""

    def __init__(self):
        self.idx = {s: torch.tensor(sorted(v), dtype=torch.long) for s, v in U.items() if v}
        self.out = {}

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        idx = self.idx.get((layer_idx, kind))
        if idx is not None:
            ta, ti = bank.encode(x, kind, layer_idx)
            dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
            self.out[(layer_idx, kind)] = dense[..., idx.to(dense.device)].float().cpu().numpy()
        return x


SITES = sorted(U); COL = {}
for s in SITES:
    for j, i in enumerate(sorted(U[s])):
        COL[(s, i)] = len(COL)
LAT = [None] * len(COL)
for (s, i), j in COL.items():
    LAT[j] = (s, i)
ROLES = ["c1", "c2", "tens", "units", "f1", "f2"]
ACT = {ro: np.zeros((len(rows), len(COL)), dtype=np.float32) for ro in ROLES}   # activation at each role position
with torch.no_grad():
    off = 0
    for b, tk in batches(rows):
        cap = CaptureU(); inference.forward(tk, patcher=cap, grad_enabled=False, return_activations=False, tokenize_final=False)
        for s, a in cap.out.items():
            cols = [COL[(s, i)] for i in sorted(U[s])]
            for j, r in enumerate(b):
                for ro in ROLES:
                    ACT[ro][off + j, cols] = a[j, r["rpos"][ro]]
        off += len(b)

# ---- 1. roles --------------------------------------------------------------------------------
W_U = inference.model.lm_head.weight.detach().float() * inference.model.transformer.norm_f.scale.detach().float()[None, :]
dig = torch.tensor(digit_ids, device=W_U.device)
DLA = np.zeros((len(COL), 10), dtype=np.float32)
for (s, i), j in COL.items():
    dvec = bank.saes[s[1]][s[0]].decoder.weight[:, i].detach().float().to(W_U.device)
    DLA[j] = (W_U[dig] @ dvec).cpu().numpy()
dla_corr = np.array([np.corrcoef(np.arange(10), DLA[j])[0, 1] if DLA[j].std() > 0 else 0 for j in range(len(COL))])
TVALS = sorted(set(tens_of))


def tens_profile(A):
    """per latent: mean act and firing rate for each tens value; detector/graded classification."""
    mean_t = np.stack([A[tens_of == t].mean(0) for t in TVALS], 1)          # [n, |T|]
    fire_t = np.stack([(A[tens_of == t] > 0).mean(0) for t in TVALS], 1)
    kind = np.array(["-"] * A.shape[1], dtype=object); best = np.full(A.shape[1], -1)
    for j in range(A.shape[1]):
        if fire_t[j].max() < 0.5:
            continue
        top = int(fire_t[j].argmax()); others = np.delete(fire_t[j], top)
        if fire_t[j, top] >= 0.7 and others.max() <= 0.2:
            kind[j] = "DETECTOR"; best[j] = TVALS[top]; continue
        ok = mean_t[j] > 0
        if ok.sum() >= 3:
            c = np.corrcoef(np.array(TVALS), mean_t[j])[0, 1]
            if abs(c) >= 0.6:
                kind[j] = "GRADED%s" % ("+" if c > 0 else "-")
    return mean_t, fire_t, kind, best


fire_rate = {ro: (ACT[ro] > 0).mean(0) for ro in ROLES}
prof = {ro: tens_profile(ACT[ro]) for ro in ("tens", "units", "f2", "f1")}
is_frame = np.zeros(len(COL), bool); is_tens = np.zeros(len(COL), bool); is_trans = np.zeros(len(COL), bool); is_out = np.zeros(len(COL), bool)
for j in range(len(COL)):
    fr = max(fire_rate["c2"][j], fire_rate["f1"][j], fire_rate["f2"][j])
    flat = prof["f2"][2][j] == "-" and prof["tens"][2][j] == "-"
    is_frame[j] = fr >= 0.9 and flat
    is_tens[j] = prof["tens"][2][j] != "-"
    is_trans[j] = prof["f2"][2][j] != "-"
    is_out[j] = fire_rate["f2"][j] >= 0.3 and abs(dla_corr[j]) >= 0.7 and np.abs(DLA[j]).max() >= 0.3
print("\n[1] ROLES over %d member latents: FRAME %d | TENS readers %d | TRANSPORT (T-dependent at final pos) %d | OUTPUT (monotone digit DLA at final pos) %d"
      % (len(COL), is_frame.sum(), is_tens.sum(), is_trans.sum(), is_out.sum()))


def name(j):
    (l, k), i = LAT[j]
    return "%d.%s.%d" % (l, k, i)


def amp(j):
    s, i = LAT[j]
    return AMP[s].get(i, 1.0)


def show_tens(title, mask, ro, top=14):
    mean_t, fire_t, kind, best = prof[ro]
    js = [j for j in np.where(mask)[0]]
    js.sort(key=lambda j: -(ACT[ro][:, j].mean() * amp(j)))
    print("\n  %s (%d; top %d by act x gain at the %s position)" % (title, len(js), min(top, len(js)), ro))
    print("  %-14s %5s %-9s  " % ("member", "gain", "kind") + " ".join("T=%d" % t for t in TVALS) + "   fires%")
    for j in js[:top]:
        print("  %-14s %5.2f %-9s  " % (name(j), amp(j), kind[j] + ("(%d)" % best[j] if best[j] >= 0 else ""))
              + " ".join("%4.1f" % v for v in mean_t[j]) + "   " + " ".join("%3.0f" % (100 * v) for v in fire_t[j]))


show_tens("TENS READERS at the tens digit", is_tens, "tens")
show_tens("TRANSPORT: T-dependent at the final position (T is 6 tokens back)", is_trans, "f2")
print("\n  FRAME recognisers (%d): top by gain x mean act, with fire rates c2/f1/f2" % is_frame.sum())
js = sorted(np.where(is_frame)[0], key=lambda j: -(max(ACT["c2"][:, j].mean(), ACT["f2"][:, j].mean()) * amp(j)))[:12]
print("  " + "; ".join("%s g%.1f %.0f/%.0f/%.0f" % (name(j), amp(j), 100 * fire_rate["c2"][j], 100 * fire_rate["f1"][j], 100 * fire_rate["f2"][j]) for j in js))
print("\n  OUTPUT members (%d): DLA over digits 0-9 at the final position, weighted by activation" % is_out.sum())
js = sorted(np.where(is_out)[0], key=lambda j: -(ACT["f2"][:, j].mean() * amp(j) * np.abs(DLA[j]).max()))[:12]
for j in js:
    print("  %-14s g%.1f act %.2f corr %+.2f | " % (name(j), amp(j), ACT["f2"][:, j].mean(), dla_corr[j]) + " ".join("%+.2f" % v for v in DLA[j]))
# above-threshold promotion: for each prompt, sum over output members of act x gain x (mean DLA(d > T) - mean DLA(d <= T))
above = np.zeros(len(rows))
for n_, r in enumerate(rows):
    t = r["tens"]; w = np.array([np.mean(DLA[j, t + 1:]) - np.mean(DLA[j, :t + 1]) for j in range(len(COL))])
    above[n_] = float((ACT["f2"][n_] * np.array([amp(j) for j in range(len(COL))]) * w)[is_out].sum())
print("  net direct effect of OUTPUT members on (digits > T) - (digits <= T): mean %+.3f (per-prompt positive on %.0f%%)" % (above.mean(), 100 * (above > 0).mean()))

# ---- 2. the tens swap ------------------------------------------------------------------------
pairs = []
by_noun = defaultdict(list)
for n_, r in enumerate(rows):
    by_noun[r["noun"]].append(n_)
order = rng.permutation(len(rows))
for n_ in order:
    r = rows[n_]
    cands = [m for m in by_noun[r["noun"]] if abs(rows[m]["tens"] - r["tens"]) >= MIN_GAP]
    if not cands:
        cands = [m for m in range(len(rows)) if abs(rows[m]["tens"] - r["tens"]) >= MIN_GAP]
    if cands:
        pairs.append((int(n_), int(rng.choice(cands))))
    if len(pairs) >= N_PAIRS:
        break
print("\n[2] TENS SWAP: %d receiver/donor pairs (|T_d - T_r| >= %d; same noun for %d)" % (len(pairs), MIN_GAP, sum(rows[a]["noun"] == rows[b]["noun"] for a, b in pairs)))


class Inject:
    """set chosen latents at chosen positions to given values (dense), per batch row: spec[b] = {site: {pos: (idx, vals)}}."""

    def __init__(self, spec):
        self.spec = spec

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        s = (layer_idx, kind)
        if not any(s in sp for sp in self.spec):
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        code = dense.clone()
        for b, sp in enumerate(self.spec):
            for p, (idx, vals) in sp.get(s, {}).items():
                code[b, p, idx.to(code.device)] = vals.to(code.device, code.dtype)
        return x + bank.decode(code - dense, kind, layer_idx, add_bias=False).to(x.dtype)


def digit_probs(rs, patch=None):
    out = []
    with torch.no_grad():
        for s0 in range(0, len(rs), EVAL_BS):
            b = rs[s0:s0 + EVAL_BS]
            L = max(len(r["ids"]) for r in b)
            tk = torch.tensor([r["ids"] + [0] * (L - len(r["ids"])) for r in b], dtype=torch.long, device=device)
            p = Inject(patch[s0:s0 + EVAL_BS]) if patch else None
            res = inference.forward(tk, patcher=p, all_logits=True, grad_enabled=False, return_activations=False, tokenize_final=False)
            lg = res[1] if isinstance(res, (tuple, list)) else res
            an = torch.tensor([len(r["ids"]) - 1 for r in b], device=device); rr = torch.arange(len(b), device=device)
            pr = torch.softmax(lg[rr, an].float(), -1)[:, dig].cpu().numpy()
            out.append(pr)
    return np.concatenate(out)


def gap_mass(P, tr, td):
    lo, hi = min(tr, td), max(tr, td)
    return float(P[lo + 1:hi + 1].sum())


def spec_for(pair, cols, roles):
    """install the donor's values of latents `cols` at the receiver's `roles` positions (donor values read at the same roles)."""
    a, b_ = pair
    sp = {}
    for ro in roles:
        by_site = defaultdict(lambda: ([], []))
        for j in cols:
            s, i = LAT[j]
            by_site[s][0].append(i); by_site[s][1].append(float(ACT[ro][b_, j]))
        for s, (ii, vv) in by_site.items():
            sp.setdefault(s, {})[rows[a]["rpos"][ro]] = (torch.tensor(ii, dtype=torch.long), torch.tensor(vv))
    return sp


recv = [rows[a] for a, _ in pairs]; donr = [rows[b_] for _, b_ in pairs]
P_r = digit_probs(recv); P_d = digit_probs(donr)
G_r = np.array([gap_mass(P_r[n_], rows[a]["tens"], rows[b_]["tens"]) for n_, (a, b_) in enumerate(pairs)])
G_d = np.array([gap_mass(P_d[n_], rows[a]["tens"], rows[b_]["tens"]) for n_, (a, b_) in enumerate(pairs)])
print("  receiver gap mass G_r = P(digit in (min,max]) mean %.3f | donor on its own prompt G_d %.3f  (swap score 1 = reaches G_d)" % (G_r.mean(), G_d.mean()))
tens_cols = list(np.where(is_tens)[0]); trans_cols = list(np.where(is_trans)[0]); all_cols = list(range(len(COL)))
firing_tens = [j for j in all_cols if fire_rate["tens"][j] > 0]
firing_f2 = [j for j in all_cols if fire_rate["f2"][j] > 0]


def rand_cols(cols, ro):
    """site-matched random latents, taken from all latents at the same sites that are not in the universe."""
    out = []
    for j in cols:
        s, _ = LAT[j]
        while True:
            i = int(rng.integers(D))
            if i not in U[s] and i not in HUBS.get(s, set()):
                break
        out.append((s, i))
    return out


def spec_rand(pair, cols, roles):
    a, b_ = pair
    sp = {}
    picks = rand_cols(cols, roles[0])
    for ro in roles:
        by_site = defaultdict(lambda: ([], []))
        for (s, i), j in zip(picks, cols):
            by_site[s][0].append(i); by_site[s][1].append(float(ACT[ro][b_, j]))
        for s, (ii, vv) in by_site.items():
            sp.setdefault(s, {})[rows[a]["rpos"][ro]] = (torch.tensor(ii, dtype=torch.long), torch.tensor(vv))
    return sp


conds = [
    ("TENS readers @ tens pos", tens_cols, ["tens"], spec_for),
    ("TENS readers @ tens+units", tens_cols, ["tens", "units"], spec_for),
    ("TRANSPORT members @ final pos", trans_cols, ["f2"], spec_for),
    ("TENS @ tens + TRANSPORT @ final", None, None, None),
    ("ALL members firing @ tens pos", firing_tens, ["tens"], spec_for),
    ("ALL members firing @ final pos", firing_f2, ["f2"], spec_for),
    ("random matched (TENS @ tens)", tens_cols, ["tens"], spec_rand),
    ("random matched (ALL @ tens)", firing_tens, ["tens"], spec_rand),
    ("random matched (ALL @ final)", firing_f2, ["f2"], spec_rand),
]
results = {}
print("\n  %-34s %6s %8s %8s %8s   %s" % ("installed from donor", "n lat", "G_swap", "score", "P(>T_d)", "argmax digit == donor-legal"))
for title, cols, roles, fn in conds:
    if cols is None:
        specs = []
        for pr in pairs:
            s1 = spec_for(pr, tens_cols, ["tens"]); s2 = spec_for(pr, trans_cols, ["f2"])
            for s, dd in s2.items():
                s1.setdefault(s, {}).update(dd)
            specs.append(s1)
        n_lat = len(tens_cols) + len(trans_cols)
    else:
        specs = [fn(pr, cols, roles) for pr in pairs]; n_lat = len(cols)
    P = digit_probs(recv, specs)
    G = np.array([gap_mass(P[n_], rows[a]["tens"], rows[b_]["tens"]) for n_, (a, b_) in enumerate(pairs)])
    score = (G - G_r) / np.where(np.abs(G_d - G_r) < 1e-6, np.nan, G_d - G_r)
    p_above_d = np.array([P[n_][rows[b_]["tens"] + 1:].sum() for n_, (_, b_) in enumerate(pairs)])
    legal = np.mean([int(P[n_].argmax()) > rows[b_]["tens"] for n_, (_, b_) in enumerate(pairs)])
    results[title] = dict(n_lat=int(n_lat), G=float(G.mean()), score=float(np.nanmedian(score)), score_mean=float(np.nanmean(score)), p_above_donor=float(p_above_d.mean()), legal=float(legal))
    print("  %-34s %6d %8.3f %8.2f %8.3f   %.0f%%" % (title, n_lat, G.mean(), np.nanmedian(score), p_above_d.mean(), 100 * legal))
p_above_r = np.array([P_r[n_][rows[b_]["tens"] + 1:].sum() for n_, (_, b_) in enumerate(pairs)])
print("  %-34s %6s %8.3f %8s %8.3f   %.0f%%" % ("(receiver, unpatched)", "-", G_r.mean(), "0.00", p_above_r.mean(),
                                                100 * np.mean([int(P_r[n_].argmax()) > rows[b_]["tens"] for n_, (_, b_) in enumerate(pairs)])))
# one worked example
a, b_ = pairs[0]
spec1 = [spec_for((a, b_), tens_cols, ["tens"])]
P1 = digit_probs([rows[a]], spec1)[0]
print("\n  EXAMPLE  receiver %r (T=%d)  donor T=%d" % (rows[a]["prompt"], rows[a]["tens"], rows[b_]["tens"]))
print("    digit        " + " ".join("%5d" % t for t in range(10)))
print("    receiver     " + " ".join("%5.2f" % v for v in P_r[0]))
print("    tens swapped " + " ".join("%5.2f" % v for v in P1))
print("    donor native " + " ".join("%5.2f" % v for v in P_d[0]))
json.dump(dict(n_universe=len(COL), roles=dict(frame=[name(j) for j in np.where(is_frame)[0]], tens=[name(j) for j in np.where(is_tens)[0]],
                                                     transport=[name(j) for j in np.where(is_trans)[0]], output=[name(j) for j in np.where(is_out)[0]]),
               tens_profiles={name(j): dict(kind=str(prof["tens"][2][j]), mean=[float(x) for x in prof["tens"][0][j]]) for j in np.where(is_tens)[0]},
               transport_profiles={name(j): dict(kind=str(prof["f2"][2][j]), mean=[float(x) for x in prof["f2"][0][j]]) for j in np.where(is_trans)[0]},
               swap=results, G_r=float(G_r.mean()), G_d=float(G_d.mean()), n_pairs=len(pairs)),
          open(R / "gt_variables.json", "w"), indent=1)
print("\n->", R / "gt_variables.json")
