"""Shared pieces of the v1 method-comparison suite (DAN-159 / DAN-43 / DAN-77): paths and env, the 062 sweep rows
(WCM and unweighted sizes and scores), ranking storage and truncation, matched-size selection, the DAN-8 pass rule,
circuit construction for the production scorer, resumable jsonl output, and the Ge bisection replay.

Pure functions only at import time (torch, numpy); nothing here loads the model or imports 062's driver, so the
tests can import it. The scripts import driver themselves, after setting OUT to the 062 run (contexts are read from
062/out_full/out/ctx, never written: all 190 sweep records exist).
"""
import glob
import json
import math
import os
import re
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent
E062 = EXP / "062-h100-protocol-v1"
RUN062 = E062 / "out_full" / "out"                       # the 15k run: contexts, main circuits, the 190-target sweep
WEAK_DIRS = ("out_bands", "out_depth", "out_lowlam")       # the weak-penalty WCM arms (paper Fig. 3): 190 targets in union
# REF_RUN (protocol v2, 2026-10-25): the WCM / unweighted references come from this run directory instead (its
# main/ and sweep/, e.g. the 8k v2 refit's OUT, whose sweep holds all 8 WCM + 5 unweighted arms); no weak-penalty
# directories are read then. It is also the driver's default OUT (a v2-marked OUT: the 083 fits run under its version).
if os.environ.get("REF_RUN"):
    RUN062, WEAK_DIRS = Path(os.environ["REF_RUN"]), ()
# Contexts come through driver.Runner.contexts, which honours the driver's CTX_DIR (DAN-157: the rebuilt held-out
# records, e.g. 062/out_full/out/ctx_v2). Outputs are kept apart per context set: OUT083 defaults to out/ for the
# as-run records and to out_<CTX_DIR name>/ otherwise, and every row carries the hash of its training contexts.
CTX_DIR = os.environ.get("CTX_DIR")
OUT = Path(os.environ.get("OUT083") or str(HERE / ("out_%s" % Path(CTX_DIR).name if CTX_DIR else "out")))
TARGETS = Path(os.environ.get("TARGETS", str(E062 / "targets_sweep.txt")))
SHARD_I, SHARD_K = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
TAG = "shard%d" % SHARD_I
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
REF_ARM = "W_0.001"                                      # the production circuit: lambda 1e-3, 400 steps (protocol v1)
# DAN-77 (curve-first): attribution is truncated at the target's own WCM size at EVERY WCM price ("W_all": the five
# sweep prices plus the three weak ones, 2.5e-5 .. 4e-3); a comma list of arm names restricts it (W_0.001 = Table 2).
SIZES = [s for s in os.environ.get("SIZES", "W_all").split(",") if s]
KEEP_TOP = int(os.environ.get("KEEP_TOP", 65536))       # ranking entries kept per (arm, ranking); the largest WCM
#                                                          circuit in the sweep is 37,170 (W_2.5e-05); 0 = every nonzero
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]    # Z, A (sparsity-preserving), C (sparsity-preserving)
NEC = "phi_sup_blind_tk"
IND = "phi_cf_alpha_blind_tk"
BAND = (0.75, 1.5)                                       # DAN-8 (floor 0.75 from 2026-10-31): Z, A, C in BAND, nec >= NEC_MIN
NEC_MIN = 0.9
DEPTH_BANDS = (("L0-4", 0, 4), ("L5-7", 5, 7), ("L8-11", 8, 11))


def import_driver():
    """062's driver (the production Runner: contexts, fit, score), pointed at the 15k run's context cache. Scoring
    uses the explicit held-out split (protocol-v1 §3.6; identical to the as-run split on held-out strongest)."""
    import sys
    os.environ.setdefault("OUT", str(RUN062))
    os.environ.setdefault("EVAL_SPLIT", "explicit")
    if str(E062) not in sys.path:
        sys.path.insert(0, str(E062))
    import driver
    return driver


def training_contexts(R, rec):
    """(arm, thin, pos, arg, hash): the fit's training rows, i.e. the first n_train rows of the probe
    driver.Runner.fit injects (circuit/protocol_contexts.training_probe), asserted equal to training_set, on R's
    device. Every 083 discovery runs on exactly these."""
    from circuit.protocol_contexts import training_probe, training_set
    arm, thin = R.train_arm(rec)
    pos, _, arg = training_set(rec, arm)
    pd_ = training_probe(rec, R.dev, arm)
    n = int(pd_.metadata["n_train"])
    assert n == int(pos.shape[0]) and torch.equal(pd_.pos_tokens[:n].cpu(), pos.cpu()), "training rows differ from the fit's"
    return arm, thin, pos.to(R.dev), arg.to(R.dev), train_hash(pos, arg)


def provenance(R, h):
    """Row fields naming the context set a row was scored on."""
    return dict(ctx_dir=str(R.ctx_dir), ctx_hash=h)


# ----------------------------------------------------------------------------------------------- targets / keys
def parse(key):
    l, k, i = key.split(".")
    return int(l), k, int(i)


def key_seed(key):
    """A per-target RNG seed (stable across processes and resumes)."""
    return zlib.crc32(key.encode()) % (2 ** 31)


def depth_band(layer):
    return next((name for name, lo, hi in DEPTH_BANDS if layer <= hi), DEPTH_BANDS[-1][0])


def shard_targets(path=None):
    keys = [t for t in Path(path or TARGETS).read_text().split() if t][SHARD_I::SHARD_K]
    return keys[:LIMIT] if LIMIT else keys


def site_order(kinds, sites):
    """Causal (forward) order of (layer, kind) sites: layer, then the bank's kind order."""
    kinds = list(kinds)
    return sorted(sites, key=lambda s: (s[0], kinds.index(s[1])))


# ----------------------------------------------------------------------------------------------- 062 sweep rows
def read_jsonl(pattern, source_root=ROOT):
    """Rows of every file matching `pattern`, each tagged with `source` = '<path relative to the repo>:<line>' so
    every table number traces to a row. Truncated lines (a kill mid-write) are skipped."""
    rows = []
    for f in sorted(glob.glob(str(pattern))):
        rel = os.path.relpath(f, source_root).replace(os.sep, "/")
        with open(f) as fh:
            for ln, line in enumerate(fh, 1):
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                r["source"] = "%s:%d" % (rel, ln)
                rows.append(r)
    return rows


def ok_rows(rows):
    return [r for r in rows if "error" not in r and "skip" not in r]


def last_by(rows, fields):
    """The last row per key (resumed runs append; merge keeps the last, as 062's merge does)."""
    out = {}
    for r in rows:
        out[tuple(r.get(f) for f in fields)] = r
    return list(out.values())


SWEEP_ARM = re.compile(r"^[WU]_[0-9.e+-]+$")             # plain price arms (drops W_0.001_s800, the depth test)


def sweep_rows(run=RUN062, weak=True):
    """The 062 sweep (held-out strongest, H100): WCM `W_<lam>` and unweighted `U_<lam>` rows, 190 targets x 10 arms,
    plus (weak=True) the weak-penalty WCM arms W_1e-4 / 5e-5 / 2.5e-5 from out_bands / out_depth / out_lowlam."""
    rows = last_by(ok_rows(read_jsonl(Path(run) / "sweep" / "eval.*.jsonl")), ("seed", "arm"))
    if weak:
        have = {(r["seed"], r["arm"]) for r in rows}
        extra = [r for d in WEAK_DIRS for r in ok_rows(read_jsonl(E062 / d / "sweep" / "eval.*.jsonl"))
                 if SWEEP_ARM.match(str(r.get("arm"))) and (r["seed"], r["arm"]) not in have]
        rows += last_by(extra, ("seed", "arm"))
    return [r for r in rows if SWEEP_ARM.match(str(r.get("arm")))]


def circuit_path(arm, key):
    """Where 062 stored a sweep circuit: out_full/out/sweep/circuits/<arm>/, else the weak-penalty run directories."""
    for base in [RUN062] + [E062 / d for d in WEAK_DIRS]:
        p = Path(base) / "sweep" / "circuits" / arm / ("%s.pt" % key)
        if p.exists():
            return p
    return None


def resolve_sizes(have, sizes=None):
    """The size arms to truncate at for one target: `have` = {arm: n}; "W_all" / "U_all" expand to every such arm
    the target has, a named arm is kept when the target has it."""
    out = []
    for s in (SIZES if sizes is None else sizes):
        if s in ("W_all", "U_all"):
            out += sorted((a for a in have if a.startswith(s[0] + "_")), key=lambda a: float(a.split("_")[1]))
        elif s in have:
            out.append(s)
    return list(dict.fromkeys(out))


def merge_reference(base_rows, local_rows, mode):
    """WCM / unweighted reference rows: mode "062" = the H100 sweep rows; "local" = only the same-machine re-scores
    (rescore_ref.py; required when the held-out sets were rebuilt); "auto" = the local re-score where one exists for
    (seed, arm), else the 062 row."""
    if mode == "062":
        return list(base_rows)
    loc = {(r["seed"], r["arm"]): r for r in local_rows}
    if mode == "local":
        return list(loc.values())
    if mode != "auto":
        raise ValueError("REF must be auto, 062 or local, got %r" % mode)
    out = [loc.get((r["seed"], r["arm"]), r) for r in base_rows]
    have = {(r["seed"], r["arm"]) for r in base_rows}
    return out + [r for k, r in loc.items() if k not in have]


def train_hash(pos, arg):
    """A short fingerprint of a target's training contexts (tokens + anchors): cached rankings, Ge points and fitted
    circuits are filed under it, so a rebuilt context set never reuses work done on another."""
    import hashlib
    h = hashlib.sha1()
    h.update(pos.detach().cpu().to(torch.int64).numpy().tobytes())
    h.update(arg.detach().cpu().to(torch.int64).numpy().tobytes())
    return h.hexdigest()[:10]


def sizes_by_arm(rows):
    """{seed: {arm: n}} from sweep rows."""
    out = defaultdict(dict)
    for r in rows:
        if r.get("n") is not None:
            out[r["seed"]][r["arm"]] = int(r["n"])
    return dict(out)


def nearest_size(n_ref, candidates):
    """The candidate arm whose size is nearest n_ref on a log scale (the D10 'nearest-size U arm per target' rule).
    Ties go to the smaller circuit, then to the arm name. `candidates` = {arm: n}; arms with n <= 0 are ignored.
    Returns None when nothing is eligible."""
    best = None
    for arm, n in candidates.items():
        if n is None or n <= 0 or n_ref is None or n_ref <= 0:
            continue
        k = (round(abs(math.log(n) - math.log(n_ref)), 9), n, arm)     # rounded: an exact log tie stays a tie
        if best is None or k < best:
            best = k
    return None if best is None else best[2]


# ----------------------------------------------------------------------------------------------- rankings
def ranking_from_site_scores(scores, sites, keep_top=KEEP_TOP, key=None):
    """A ranking from dense per-site score vectors: every nonzero (site, latent), sorted by |score| descending,
    ties by site position in `sites` (causal order) then latent index, so the order is deterministic.
    `scores` = {site: [d_sae] tensor}; `key` (optional) = {site: [d_sae] tensor} ranked by instead of |score|
    (descending). Returns {"site": int16, "latent": int32, "score": float32} numpy arrays, truncated to keep_top
    (0 = keep all)."""
    si, la, sc, kk = [], [], [], []
    for j, s in enumerate(sites):
        v = scores.get(s)
        if v is None:
            continue
        v = v.detach().float().cpu()
        k = (key[s].detach().float().cpu() if key is not None else v.abs())
        nz = (v != 0).nonzero(as_tuple=True)[0]
        if keep_top and nz.numel() > keep_top:                 # a site cannot contribute more than keep_top entries
            top = k[nz].topk(keep_top).indices
            nz = nz[top].sort().values
        si.append(np.full(nz.numel(), j, dtype=np.int16))
        la.append(nz.numpy().astype(np.int32))
        sc.append(v[nz].numpy().astype(np.float32))
        kk.append(k[nz].numpy().astype(np.float64))
    if not si:
        return empty_ranking()
    si, la, sc, kk = (np.concatenate(x) for x in (si, la, sc, kk))
    order = np.lexsort((la, si, -kk))
    if keep_top:
        order = order[:keep_top]
    return {"site": si[order], "latent": la[order], "score": sc[order]}


def ranking_from_items(items, sites, keep_top=KEEP_TOP):
    """A ranking from an explicit order: `items` = [(site, latent, score)], already in rank order."""
    idx = {s: j for j, s in enumerate(sites)}
    items = list(items)[:keep_top] if keep_top else list(items)
    if not items:
        return empty_ranking()
    return {"site": np.array([idx[s] for s, _, _ in items], dtype=np.int16),
            "latent": np.array([int(i) for _, i, _ in items], dtype=np.int32),
            "score": np.array([float(v) for _, _, v in items], dtype=np.float32)}


def empty_ranking():
    return {"site": np.zeros(0, np.int16), "latent": np.zeros(0, np.int32), "score": np.zeros(0, np.float32)}


def truncate_global(rank, sites, n):
    """The first n entries of a ranking as {site: [latents]} (the Table 2 top-n rows: global top-n, matched to the
    reference circuit's size). Fewer than n ranked entries -> all of them."""
    n = max(0, int(n))
    out = defaultdict(list)
    for j, i in zip(rank["site"][:n].tolist(), rank["latent"][:n].tolist()):
        out[sites[j]].append(int(i))
    return dict(out)


def truncate_per_site(rank, sites, counts):
    """The top counts[site] entries of each site, in rank order (DAN-43: attribution members at the production
    circuit's per-site counts). Returns ({site: [latents]}, short) where `short` lists the sites whose ranking held
    fewer entries than asked for (all of them are taken, as random_null.py does with too few live latents)."""
    want = {s: int(c) for s, c in counts.items() if int(c) > 0}
    out = defaultdict(list)
    for j, i in zip(rank["site"].tolist(), rank["latent"].tolist()):
        s = sites[j]
        if len(out[s]) < want.get(s, 0):
            out[s].append(int(i))
    short = sorted(s for s, c in want.items() if len(out.get(s, ())) < c)
    return {s: v for s, v in out.items() if v}, short


def n_members(members):
    return sum(len(v) for v in members.values())


def save_ranking_file(path, payload):
    """Atomic save (tmp + rename), so an interrupted write never leaves a half file that resume would trust."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- reductions
class AttrReducer:
    """Per-latent reductions of one site's per-position attribution, accumulated chunk by chunk:
      max  the signed value at the largest |attr| over every (context, position <= anchor)  (031's ranking:
           "its largest score at any context and position")
      agg  mean over contexts of the summed |attr| over the causal prefix  (the plan's mean |attr| ranking)
      sfc  mean over contexts of the summed SIGNED attr over the prefix, i.e. the classic collapsed IG score
           (sparse feature circuits' aggregate; equals integrated_baseline_scores without position_aware)
    `update` takes PER-SEQUENCE attributions [B, T, d] (each sequence's attribution to its own anchor read) and the
    anchors [B]; positions after the anchor are masked (they carry no gradient anyway: the model is causal)."""

    def __init__(self, d, device="cpu"):
        self.max = torch.zeros(d, dtype=torch.float32, device=device)
        self.abs_sum = torch.zeros(d, dtype=torch.float64, device=device)
        self.sum = torch.zeros(d, dtype=torch.float64, device=device)
        self.n = 0

    def update(self, attr, anchors):
        B, T, d = attr.shape
        anc = anchors[:B].to(attr.device).long().clamp(0, T - 1)
        keep = (torch.arange(T, device=attr.device)[None, :] <= anc[:, None]).to(attr.dtype)    # [B, T]
        a = (attr * keep[..., None]).float()
        flat = a.reshape(B * T, d)
        idx = flat.abs().argmax(dim=0)                          # first maximal (context, position) per latent
        cand = flat.gather(0, idx[None]).squeeze(0)
        self.max = torch.where(cand.abs() > self.max.abs(), cand, self.max.to(cand.device))   # earlier chunk wins ties
        self.abs_sum += a.abs().sum(dim=(0, 1)).double()
        self.sum += a.sum(dim=(0, 1)).double()
        self.n += B

    def result(self):
        n = max(self.n, 1)
        return {"max": self.max, "agg": (self.abs_sum / n).float(), "sfc": (self.sum / n).float()}


# ----------------------------------------------------------------------------------------------- circuits / rows
def make_circuit(key, members, amps=None, name="v1cmp"):
    """A Circuit the production scorer reads: a seed node plus one node per member; `amps` = {site: {latent: alpha}}
    (absent -> the scorer applies alpha = 1, as for unweighted circuits)."""
    from circuit.types.feature_id import FeatureID
    from store.circuits import Circuit, CircuitNode
    l, k, i = parse(key)
    c = Circuit(name="%s_%s" % (name, key))
    c.add_node(CircuitNode(metadata={"feature_id": FeatureID(l, k, i), "role": "seed"}))
    for (sl, sk), idx in members.items():
        for j in idx:
            md = {"feature_id": FeatureID(int(sl), str(sk), int(j)), "role": "ablation_support"}
            a = (amps or {}).get((sl, sk), {}).get(int(j))
            if a is not None:
                md["amplitude"] = float(a)
            c.add_node(CircuitNode(metadata=md))
    return c


def members_by_site(c):
    """{(layer, kind): [latents]} of a stored circuit (seed excluded)."""
    out = defaultdict(list)
    for nd in c.nodes.values():
        md = nd.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out[(int(f.layer), str(f.kind))].append(int(f.index))
    return dict(out)


def scalar_row(r):
    """Keep the JSON-able scalar fields of a scorer row (per_kind etc. are dicts of ints: kept as well)."""
    out = {}
    for k, v in r.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict) and all(isinstance(x, (int, float, str, bool)) or x is None for x in v.values()):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v) and len(v) <= 64:
            out[k] = list(v)
    return out


def done_keys(path, fields):
    """Finished keys from EVERY shard's file of this kind (rows.shard*.jsonl), so a rerun with another shard count
    skips work any shard finished. Error rows are not done (they are retried)."""
    path = Path(path)
    out = set()
    for p in sorted(path.parent.glob(path.name.replace(TAG, "shard*"))):
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if "error" not in r:
                    out.add(tuple(r.get(f) for f in fields))
    return out


def write_row(fh, row):
    fh.write(json.dumps(row) + "\n")
    fh.flush()


# ----------------------------------------------------------------------------------------------- pass rule
VAC_BASE = ("e0_tk", "eM_topk_tk", "eN_topk_tk")   # the pass rule's denominators: Z and necessity, A, C


def vacuous(r):
    """Protocol §12.4, as src/eval/wcm_eval.is_vacuous: a_pos_tk <= 0 or any VAC_BASE baseline above 0.95 a_pos_tk;
    the stored vacuous_v2 when present, the v1 flag vacuous_tk for a row without those reads."""
    if r.get("vacuous_v2") is not None:
        return bool(r["vacuous_v2"])
    ap, es = r.get("a_pos_tk"), [r.get(k) for k in VAC_BASE]
    if ap is None or any(e is None for e in es):
        return bool(r.get("vacuous_tk"))
    ap = float(ap)
    return bool(ap <= 0 or any(ap - float(e) < 0.05 * ap for e in es))


def passes(r):
    """DAN-8 on the activation read: Z, A, C all in [0.75, 1.5] and necessity >= 0.9; a missing score fails.
    Apply to non-vacuous rows (the vacuous denominator is excluded and counted, as 062/merge.py does)."""
    for c in HEAD:
        v = r.get(c)
        if v is None or not (BAND[0] <= float(v) <= BAND[1]):
            return False
    nec = r.get(NEC)
    return nec is not None and float(nec) >= NEC_MIN


# ----------------------------------------------------------------------------------------------- Ge bisection
GE_LO, GE_HI = 1e-8, 1e2                                 # 034's log-space bracket


def ge_plan(points, n_target, steps, tol, lo=GE_LO, hi=GE_HI):
    """034's discover_at_size as a replayable plan, so an interrupted target resumes mid-bisection: given the points
    already run ([{"tau", "n"}], the first at tau = 0), return ("run", tau) for the next sweep or ("done", index) of
    the point the bisection returns (the first within tol of n_target, else the closest, else tau = 0 when the
    all-active set is already no larger than the target)."""
    if not points:
        return ("run", 0.0)
    if points[0]["n"] <= n_target:
        return ("done", 0)
    best = 0
    for it in range(steps):
        mid = (lo * hi) ** 0.5
        if len(points) <= it + 1:
            return ("run", mid)
        p = points[it + 1]
        if not math.isclose(p["tau"], mid, rel_tol=1e-9):
            raise ValueError("Ge point %d has tau %.6g, the replay expects %.6g (stale points file?)" % (it + 1, p["tau"], mid))
        n = p["n"]
        if abs(n - n_target) < abs(points[best]["n"] - n_target):
            best = it + 1
        if abs(n - n_target) <= tol * n_target:
            return ("done", it + 1)
        if n > n_target:
            lo = mid
        else:
            hi = mid
    return ("done", best)
