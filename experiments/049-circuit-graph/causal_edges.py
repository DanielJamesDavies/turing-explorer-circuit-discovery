"""CIRCUIT-LEVEL CAUSAL EFFECTS (GPU, local RTX) on the production circuits.

Item 4 of Daniel's list (2026-09-07): are the seed->seed membership edges
CAUSAL? For an edge A->B (A's seed latent is a member of B's circuit),
measure B's seed activation on B's own positive probes under:

  full         nothing ablated
  seedA        A's seed latent alone zeroed (the single-latent edge)
  circA        A's whole circuit zeroed (seed + members, at their sites)
  circA_nosh   A's circuit minus the latents that are ALSO B's members
               (the part of A that reaches B only indirectly)
  ctrl         a matched NON-edge circuit C zeroed (same seed layer, size
               within 30% of A, C's seed not a member of B)
  rand         random live latents matched to A's per-site counts

effect = (a_full - a_ablated) / a_full. Edge vs ctrl/rand answers "is the
membership edge a causal dependency, over and above ablating ~|A| latents".

Item 5: JACOBIAN of B's seed w.r.t. A's seed latent code, summed over
positions and multiplied by A's natural activation = the FIRST-ORDER
prediction of the seedA ablation; compared with the measured seedA effect.

Item 4b (family cores): for families with a shared core (families_nohub),
zero the core on member circuits' probes vs a matched random set.

  N_EDGES=120 N_FAM=8 PER_FAM=12 PYTHONPATH=src python experiments/049-circuit-graph/causal_edges.py
SMOKE=1 -> 3 edges, 1 family.
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import component_idx as comp_of
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense

HERE = Path(__file__).parent
T = Path(os.environ.get("TABLES", str(HERE / "tables")))
R = Path(os.environ.get("OUT", str(HERE / "results")))
SMOKE = os.environ.get("SMOKE") == "1"
# Sources at or above this layer only. The snapshot run (sources L0-L3
# only) showed L0 token-feature sources carry no specific dependency:
# their whole effect is the hubs they contain. Mid-layer sources are the
# real test, and they only exist in the full set.
SRC_MIN_LAYER = int(os.environ.get("SRC_MIN_LAYER", 0))
TAG = os.environ.get("TAG", "")
N_EDGES = 3 if SMOKE else int(os.environ.get("N_EDGES", 120))
N_FAM = 1 if SMOKE else int(os.environ.get("N_FAM", 8))
PER_FAM = 3 if SMOKE else int(os.environ.get("PER_FAM", 12))
N_SEQ, EVAL_BS = 64, 16
SEED = int(os.environ.get("SEED", 0))
random.seed(SEED); np.random.seed(SEED)

torch.set_float32_matmul_precision("high")
load_discovery_artifacts("outputs", candidates_path="outputs/candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
KINDS = list(bank.kinds); NK = len(KINDS); D = bank.d_sae
avg_acts = torch.zeros((bank.n_layer * NK, D), device=bank.device)
disc = config.discovery
disc.probe_sequence_count = N_SEQ; disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS; disc.probe_batch_size = 4; disc.position_aware = False
for p in inference.model.parameters():
    p.requires_grad_(False)
M0 = _build_mode_method("counterfactual_gradient", "local", inference, bank, avg_acts, pb)

# ---- tables ------------------------------------------------------------------
C = pd.read_parquet(T / "circuits.parquet")
M = pd.read_parquet(T / "members.parquet")
E = pd.read_csv(R / "seed_seed_edges.csv")
seed_of = {int(r.cid): (int(r.seed_layer), str(r.seed_kind), int(r.seed_index)) for r in C.itertuples()}
size_of = dict(zip(C["cid"].astype(int), C["n_members"].astype(int)))
members = defaultdict(lambda: defaultdict(set))          # cid -> site -> {idx}
amps = defaultdict(lambda: defaultdict(dict))            # cid -> site -> {idx: amplitude}
for cid, lay, knd, idx, am in zip(M["cid"].values, M["layer"].values, M["kind"].astype(str).values, M["index"].values, M["amplitude"].values):
    members[int(cid)][(int(lay), knd)].add(int(idx))
    amps[int(cid)][(int(lay), knd)][int(idx)] = float(am) if am == am else 1.0
# hubs (fan-out >= 5% of circuits): the universal L0/L1 infrastructure. Any
# circuit containing them hurts everything when removed, so every edge is
# also tested with hubs EXCLUDED from the ablated set.
HUB_FRAC = float(os.environ.get("HUB_FRAC", 0.05))
_fo = pd.read_csv(R / "fanout_latents.csv")
HUBS = defaultdict(set)
for _, r in _fo[_fo["fanout"] >= HUB_FRAC * len(C)].iterrows():
    HUBS[(int(r["layer"]), str(r["kind"]))].add(int(r["index"]))
print("hubs excluded in *_nohub conditions: %d" % sum(len(v) for v in HUBS.values()), flush=True)


def drop_hubs(sets):
    return {st: v - HUBS.get(st, set()) for st, v in sets.items()}
live_pool = defaultdict(list)                             # site -> sorted live latents
for site, s in ((k, v) for d in members.values() for k, v in d.items()):
    live_pool[site].extend(s)
live_pool = {k: np.array(sorted(set(v))) for k, v in live_pool.items()}
print("tables: %d circuits, %d edges" % (len(C), len(E)), flush=True)


def circuit_set(cid, include_seed=True):
    s = {k: set(v) for k, v in members[cid].items()}
    if include_seed:
        l, k, i = seed_of[cid]
        s.setdefault((l, k), set()).add(i)
    return s


def matched_random(sets, exclude=None):
    out = {}
    for site, s in sets.items():
        pool = live_pool.get(site)
        if pool is None or len(pool) == 0:
            continue
        ex = (exclude or {}).get(site, set()) | s
        cand = pool[~np.isin(pool, list(ex))] if ex else pool
        n = min(len(s), len(cand))
        out[site] = set(np.random.choice(cand, n, replace=False).tolist()) if n else set()
    return out


class AblateSetPatcher:
    """Zero the given {site: {idx}} latents in the live stream (error term
    untouched); tap B's seed pre-activation at its site."""

    def __init__(self, sets, seed_site, w_seed, b_seed, state=None):
        self.sets = {s: torch.tensor(sorted(v), dtype=torch.long) for s, v in sets.items() if v}
        self.seed_site = seed_site; self.w_seed = w_seed; self.b_seed = b_seed
        self.seed_pre = None
        # B-as-a-unit readout: {site: {idx: amplitude}} of B's own members;
        # amplitude-weighted activation summed over positions, read from the
        # stream AFTER the edit at that site (what downstream sees).
        self.state = {st: (torch.tensor(sorted(d), dtype=torch.long),
                           torch.tensor([d[i] for i in sorted(d)], dtype=torch.float32))
                      for st, d in (state or {}).items() if d}
        self.state_sum = 0.0

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            w = self.w_seed.to(device=x.device, dtype=x.dtype); b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b
            return x
        idx = self.sets.get((layer_idx, kind))
        st = self.state.get((layer_idx, kind))
        if idx is None and st is None:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        out, code = x, dense
        if idx is not None:
            code = dense.clone(); code[..., idx.to(dense.device)] = 0
            out = x + bank.decode(code - dense, kind, layer_idx, add_bias=False).to(x.dtype)
        if st is not None:
            si, sa = st[0].to(dense.device), st[1].to(dense.device)
            self.state_sum += float((code[..., si].float() * sa).sum())
        return out


class JacobianPatcher:
    """Adds delta[b,t] * W_dec[a_idx] at A's site through the SAE decode
    (delta is a leaf) and taps B's seed; also records A's natural code."""

    def __init__(self, a_site, a_idx, seed_site, w_seed, b_seed):
        self.a_site, self.a_idx = a_site, a_idx
        self.seed_site = seed_site; self.w_seed = w_seed; self.b_seed = b_seed
        self.seed_pre = None; self.delta = None; self.a_act = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            w = self.w_seed.to(device=x.device, dtype=x.dtype); b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b
            return x
        if (layer_idx, kind) != self.a_site:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        self.a_act = dense[..., self.a_idx].detach().float()
        self.delta = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=torch.float32, requires_grad=True)
        code = torch.zeros_like(dense); code[..., self.a_idx] = self.delta.to(dense.dtype)
        return x + bank.decode(code, kind, layer_idx, add_bias=False).to(x.dtype)


def read(patcher, tokens, anchors, grad=False):
    """mean relu(seed_pre at anchor); with grad=True returns per-batch (pred, meas) for the Jacobian."""
    tot, n, jac_pred = 0.0, 0, 0.0
    if hasattr(patcher, "state_sum"):
        patcher.state_sum = 0.0
    inference.disable_compile()
    try:
        for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
            tk = tokens[s0:s0 + EVAL_BS]
            patcher.seed_pre = None
            inference.forward(tk, patcher=patcher, grad_enabled=grad, return_activations=False, tokenize_final=False)
            pre = patcher.seed_pre; B = pre.shape[0]
            rr = torch.arange(B, device=pre.device)
            anc = anchors[s0:s0 + B].to(pre.device).clamp(0, pre.shape[1] - 1)
            v = pre[rr, anc]
            if grad:
                v.float().sum().backward()
                J = patcher.delta.grad                     # d seed_pre / d code_A  [B, T]
                jac_pred += float((J * patcher.a_act).sum())   # first-order drop if A's code -> 0
                patcher.delta = None
            tot += float(torch.relu(v.detach()).sum()); n += B
    finally:
        inference.enable_compile()
    if grad:
        return tot / max(n, 1), jac_pred / max(n, 1)
    return tot / max(n, 1)


def read2(patcher, tokens, anchors):
    """(seed activation, amplitude-weighted B-state) under one patcher."""
    a = read(patcher, tokens, anchors)
    return a, float(getattr(patcher, "state_sum", 0.0)) / max(int(tokens.shape[0]), 1)


def read_all(patcher, tokens, anchors, grad=False):
    """Per-probe relu(seed_pre at anchor) [N] plus the B-state mean; with
    grad=True also the Jacobian first-order drop and an A-ACTIVE mask [N]
    (A's natural code non-zero anywhere in the probe). The conditional
    effect (probes where A is live) is the fair test for a source that
    fires on a few of B's contexts."""
    vals, act, jac_pred = [], [], 0.0
    if hasattr(patcher, "state_sum"):
        patcher.state_sum = 0.0
    inference.disable_compile()
    try:
        for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
            tk = tokens[s0:s0 + EVAL_BS]
            patcher.seed_pre = None
            inference.forward(tk, patcher=patcher, grad_enabled=grad, return_activations=False, tokenize_final=False)
            pre = patcher.seed_pre; B = pre.shape[0]
            rr = torch.arange(B, device=pre.device)
            anc = anchors[s0:s0 + B].to(pre.device).clamp(0, pre.shape[1] - 1)
            v = pre[rr, anc]
            if grad:
                v.float().sum().backward()
                J = patcher.delta.grad
                jac_pred += float((J * patcher.a_act).sum())
                act.append((patcher.a_act.amax(dim=1) > 0).cpu())
                patcher.delta = None
            vals.append(torch.relu(v.detach()).float().cpu())
    finally:
        inference.enable_compile()
    n = max(int(tokens.shape[0]), 1)
    vec = torch.cat(vals)
    state = float(getattr(patcher, "state_sum", 0.0)) / n
    if grad:
        return vec, state, jac_pred / n, torch.cat(act)
    return vec, state


def eff_vec(full, v, mask=None):
    """Fractional drop of the mean seed activation; with mask, over the
    masked probes only (nan when fewer than 4)."""
    if mask is not None:
        if int(mask.sum()) < 4:
            return float("nan")
        full, v = full[mask], v[mask]
    f = float(full.mean())
    return (f - float(v.mean())) / f if abs(f) > 1e-6 else float("nan")


def probes_for(cid):
    l, k, i = seed_of[cid]
    pd_ = M0.build_probe_dataset(comp_of(l, KINDS.index(k), NK), i)
    if pd_ is None or int(pd_.pos_tokens.shape[0]) < 16:
        return None
    sae = bank.saes[k][l]
    return dict(pt=pd_.pos_tokens[:N_SEQ], pa=pd_.pos_argmax[:N_SEQ], site=(l, k), idx=i,
                w=sae.encoder.weight[i].detach(), b=sae._get_bias_eff()[i].detach())


def effect(a_full, a):
    return (a_full - a) / a_full if abs(a_full) > 1e-6 else float("nan")


# ==== PART 1: edges ============================================================
lay = C.set_index("cid")["seed_layer"]
E = E[(lay.loc[E["src"]].values < lay.loc[E["dst"]].values) & (lay.loc[E["src"]].values >= SRC_MIN_LAYER)]
E = E.sample(frac=1.0, random_state=SEED)
seen_dst = defaultdict(int); picked = []
for r in E.itertuples():
    if seen_dst[int(r.dst)] >= 2:
        continue
    seen_dst[int(r.dst)] += 1; picked.append((int(r.src), int(r.dst)))
    if len(picked) >= N_EDGES:
        break
print("edges picked: %d (B layers %s)" % (len(picked), sorted(set(lay.loc[[b for _, b in picked]].tolist()))), flush=True)
edge_targets = defaultdict(set)
for a, b in E[["src", "dst"]].itertuples(index=False):
    edge_targets[int(b)].add(int(a))
by_layer_size = defaultdict(list)
for cid in C["cid"].astype(int):
    by_layer_size[seed_of[cid][0]].append(cid)

out_path = R / ("causal_edges%s%s.jsonl" % ("_smoke" if SMOKE else "", TAG))
fh = open(out_path, "w")
t0 = time.time(); cache = {}
for n_done, (a, b) in enumerate(picked):
    if b not in cache:
        cache = {b: probes_for(b)}
    P = cache[b]
    if P is None:
        continue
    A = circuit_set(a); Bm = circuit_set(b, include_seed=False)
    A_nosh = {s: v - Bm.get(s, set()) for s, v in A.items()}
    shared = sum(len(v & Bm.get(s, set())) for s, v in A.items())
    # control circuit: same seed layer as A, size within 30%, not an edge into B
    la = seed_of[a][0]; sa = size_of[a]
    pool = [c for c in by_layer_size[la] if c != a and c not in edge_targets[b] and abs(size_of[c] - sa) <= 0.3 * max(sa, 10)
            and seed_of[c][2] not in Bm.get(seed_of[c][:2], set())]
    ctrl = random.choice(pool) if pool else None
    Cs = circuit_set(ctrl) if ctrl is not None else None
    Rs = matched_random(A, exclude=Bm)
    Bst = {st: dict(d) for st, d in amps[b].items()}
    mk = lambda sets: AblateSetPatcher(sets, P["site"], P["w"], P["b"], state=Bst)
    # Jacobian pass first: it also yields the A-active mask over B's probes
    jp = JacobianPatcher(seed_of[a][:2], seed_of[a][2], P["site"], P["w"], P["b"])
    _, _, jac_pred, a_on = read_all(jp, P["pt"], P["pa"], grad=True)
    n_active = int(a_on.sum())
    V = {}; S = {}
    A_nh, C_nh = drop_hubs(A), (drop_hubs(Cs) if Cs is not None else None)
    conds = {"full": {}, "seedA": {seed_of[a][:2]: {seed_of[a][2]}}, "circA": A, "nosh": A_nosh, "ctrl": Cs, "rand": Rs,
             "circA_nh": A_nh, "ctrl_nh": C_nh, "rand_nh": matched_random(A_nh, exclude=Bm)}
    for name, sets in conds.items():
        if sets is None:
            V[name], S[name] = None, float("nan")
            continue
        V[name], S[name] = read_all(mk(sets), P["pt"], P["pa"])
    vf = V["full"]
    a_full, s_full = float(vf.mean()), S["full"]
    E_ = lambda k: eff_vec(vf, V[k]) if V[k] is not None else float("nan")          # over all probes
    Ec = lambda k: eff_vec(vf, V[k], a_on) if V[k] is not None else float("nan")   # over A-active probes
    a_seed = float(V["seedA"].mean())
    n_hub_A = sum(len(v & HUBS.get(st, set())) for st, v in A.items())
    row = dict(src=a, dst=b, src_seed="%d.%s.%d" % seed_of[a], dst_seed="%d.%s.%d" % seed_of[b],
               size_A=sa, size_B=size_of[b], shared_AB=shared, ctrl=ctrl, size_ctrl=size_of[ctrl] if ctrl is not None else None,
               ctrl_shared=(sum(len(v & Bm.get(s, set())) for s, v in Cs.items()) if Cs else None),
               a_full=a_full, eff_seedA=E_("seedA"), eff_circA=E_("circA"), eff_circA_nosh=E_("nosh"),
               eff_ctrl=E_("ctrl"), eff_rand=E_("rand"),
               n_hub_A=n_hub_A, eff_circA_nohub=E_("circA_nh"), eff_ctrl_nohub=E_("ctrl_nh"), eff_rand_nohub=E_("rand_nh"),
               n_active=n_active, effc_seedA=Ec("seedA"), effc_circA=Ec("circA"), effc_ctrl=Ec("ctrl"),
               effc_circA_nohub=Ec("circA_nh"), effc_ctrl_nohub=Ec("ctrl_nh"), effc_rand_nohub=Ec("rand_nh"),
               state_full=s_full, st_seedA=effect(s_full, S["seedA"]), st_circA=effect(s_full, S["circA"]), st_circA_nosh=effect(s_full, S["nosh"]),
               st_ctrl=effect(s_full, S["ctrl"]), st_rand=effect(s_full, S["rand"]),
               st_circA_nohub=effect(s_full, S["circA_nh"]), st_ctrl_nohub=effect(s_full, S["ctrl_nh"]), st_rand_nohub=effect(s_full, S["rand_nh"]),
               jac_pred_drop=jac_pred, meas_seedA_drop=a_full - a_seed)
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("[%3d/%d] %s -> %s | |A| %4d (hubs %3d) shared %3d | A active on %2d/%d | full %.3f | ALL: seedA %+.3f circA %+.3f ctrl %+.3f | NOHUB circA %+.3f ctrl %+.3f rand %+.3f | A-ACTIVE: seedA %+.3f circA_nh %+.3f ctrl_nh %+.3f | B-state nohub %+.3f ctrl %+.3f | jac %.3f/%.3f | %.0fs"
          % (n_done + 1, len(picked), row["src_seed"], row["dst_seed"], sa, n_hub_A, shared, n_active, len(vf), a_full, row["eff_seedA"], row["eff_circA"],
             row["eff_ctrl"], row["eff_circA_nohub"], row["eff_ctrl_nohub"], row["eff_rand_nohub"],
             row["effc_seedA"], row["effc_circA_nohub"], row["effc_ctrl_nohub"],
             row["st_circA_nohub"], row["st_ctrl_nohub"], jac_pred, row["meas_seedA_drop"], time.time() - t0), flush=True)
fh.close()

rows = [json.loads(l) for l in open(out_path)]
if rows:
    df = pd.DataFrame(rows)
    print("\nEDGE SUMMARY over %d edges (effect = fractional drop of B's seed):" % len(df))
    cols = [c for c in df.columns if c.startswith(("eff_", "effc_", "st_"))]
    for c in cols:
        v = df[c].dropna()
        if len(v):
            print("  %-17s median %+.3f | mean %+.3f | frac > 0.1: %.2f | frac > 0.5: %.2f | n=%d" % (c, v.median(), v.mean(), (v > 0.1).mean(), (v > 0.5).mean(), len(v)))
    if "n_active" in df:
        print("  A active on B's probes: median %d of 64 | edges with >= 4 active: %d" % (df["n_active"].median(), int((df["n_active"] >= 4).sum())))
    for cond, ctrl in (("eff_circA", "eff_ctrl"), ("eff_circA_nohub", "eff_ctrl_nohub"), ("effc_circA_nohub", "effc_ctrl_nohub"),
                       ("effc_seedA", "effc_rand_nohub"), ("st_circA_nohub", "st_ctrl_nohub")):
        if cond in df and ctrl in df:
            d = (df[cond] - df[ctrl]).dropna()
            if len(d):
                print("  %s minus %s: paired median %+.3f | edge > ctrl in %.2f of edges | n=%d" % (cond, ctrl, d.median(), (d > 0).mean(), len(d)))
    okc = df[["eff_circA_nohub", "st_circA_nohub"]].dropna()
    if len(okc) > 2:
        print("  B-endpoint vs B-state drop (nohub): spearman %.3f | state>0.2 & endpoint<0.1 (shortcut): %d | endpoint>0.2 & state<0.1 (bypass): %d"
              % (okc.corr(method="spearman").iloc[0, 1], int(((okc["st_circA_nohub"] > 0.2) & (okc["eff_circA_nohub"] < 0.1)).sum()),
                 int(((okc["eff_circA_nohub"] > 0.2) & (okc["st_circA_nohub"] < 0.1)).sum())))
    if "n_hub_A" in df:
        by = df.assign(Blayer=[int(x.split(".")[0]) for x in df["dst_seed"]]).groupby("Blayer")[["eff_circA", "eff_ctrl", "eff_circA_nohub", "eff_ctrl_nohub"]].median()
        print("  by B layer (medians):"); print(by.round(3).to_string())
    ok = df[["jac_pred_drop", "meas_seedA_drop"]].dropna()
    if len(ok) > 2:
        print("  Jacobian first-order prediction vs measured seedA drop: pearson %.3f | spearman %.3f | median ratio pred/meas %.2f"
              % (ok.corr().iloc[0, 1], ok.corr(method="spearman").iloc[0, 1], (ok["jac_pred_drop"] / ok["meas_seedA_drop"].replace(0, np.nan)).median()))
    print("  edge circA vs ctrl: paired median difference %+.3f | edges where circA > ctrl: %.2f"
          % ((df["eff_circA"] - df["eff_ctrl"]).median(), (df["eff_circA"] > df["eff_ctrl"]).mean()))

# ==== PART 2: family cores =====================================================
fam_path = R / "families_nohub.json"
if fam_path.exists():
    fams_all = [f for f in json.load(open(fam_path)) if f["n_core"] >= 3]
    # FAMS=23,24,26 -> test these family ids (e.g. the ones whose cores
    # replicate across halves and co-fire); default = the N_FAM largest.
    if os.environ.get("FAMS"):
        want = {int(x) for x in os.environ["FAMS"].split(",") if x}
        fams = [f for f in fams_all if f["family"] in want]
    else:
        fams = fams_all[:N_FAM]
    famcsv = pd.read_csv(R / "circuit_families_nohub.csv")
    outf = open(R / ("causal_families%s%s.jsonl" % ("_smoke" if SMOKE else "", TAG)), "w")
    print("\nFAMILY CORES: %d families with >= 3 core latents" % len(fams), flush=True)
    for f in fams:
        core = defaultdict(set)
        for s in f["core"]:
            l, k, i = s.split("."); core[(int(l), k)].add(int(i))
        max_core_layer = max(l for l, _ in core)
        cands = famcsv[(famcsv["family"] == f["family"]) & (famcsv["seed_layer"] > max_core_layer)]["cid"].astype(int).tolist()
        random.shuffle(cands)
        effs, rands = [], []
        for cid in cands[:PER_FAM]:
            P = probes_for(cid)
            if P is None:
                continue
            Bm = circuit_set(cid, include_seed=False)
            mk = lambda sets: AblateSetPatcher(sets, P["site"], P["w"], P["b"])
            a_full = read(mk({}), P["pt"], P["pa"])
            a_core = read(mk(drop_hubs(core)), P["pt"], P["pa"])
            a_rand = read(mk(matched_random(drop_hubs(core), exclude=Bm)), P["pt"], P["pa"])
            e_c, e_r = effect(a_full, a_core), effect(a_full, a_rand)
            effs.append(e_c); rands.append(e_r)
            outf.write(json.dumps(dict(family=f["family"], cid=cid, seed="%d.%s.%d" % seed_of[cid], a_full=a_full, eff_core=e_c, eff_rand=e_r,
                                       core_in_B=sum(len(v & Bm.get(s, set())) for s, v in core.items()))) + "\n"); outf.flush()
        if effs:
            print("  family %3d (%d circuits, core %d latents %s): core-ablation effect median %+.3f (rand %+.3f) | frac > 0.2: %.2f (rand %.2f) | n=%d"
                  % (f["family"], f["n"], f["n_core"], " ".join(f["core"][:4]), float(np.nanmedian(effs)), float(np.nanmedian(rands)),
                     float(np.mean(np.array(effs) > 0.2)), float(np.mean(np.array(rands) > 0.2)), len(effs)), flush=True)
    outf.close()
print("\nDONE %.0fs" % (time.time() - t0))
