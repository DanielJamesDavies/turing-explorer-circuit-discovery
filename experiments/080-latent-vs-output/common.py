"""Experiment 080 (latent- vs output-targeted circuits): the pure logic, with no model and no GPU. Shared by
make_targets.py, run.py, calibrate.py, report.py and figure.py, and unit-tested by test_common.py.

  tiers          every target is an INTERNAL latent; the tier axis is how much it affects the next-token prediction.
                 causal (default, 2026-10-18): "high" / "low next-token effect" = the top / bottom tertile of the
                 measured in-context effect (next_token_effect.py: effect_abs, the mean |delta log p(actual next token)|
                 when the target is zero-ablated at its own site) WITHIN each kind x band group's measured pool.
                 legacy (withinkind / tierband): "direct-path (064) high / mid / low" from 064's context-free boost_peak
                 (l_max x peak activation, in nats): dp_high >= 1.0, dp_low < 0.1, dp_mid in between
  bands          depth bands L0-4 / L5-7 / L8-11 (as 061 / 062 / 071)
  sampling       DEFAULT causal: one layer-matched high-vs-low pair per kind x band group (9 pairs, 9 per tier).
                 withinkind: six within-kind, within-band pairs of adjacent direct-path tiers (PAIRS), 13 per cell.
                 tierband: the original 50-per-direct-path-tier design. Both legacy samplings stay reproducible.
  P1 statistics  pair_contrast (within-pair tier differences, pooled as a simple mean; primary) and fe_slope (slope
                 on log of the tier measure with kind x band fixed effects; secondary)
  arms           LAT (the 062 production circuit, not refitted), OUT_W (WCM on the output endpoint), OUT_U (the same
                 with alpha fixed at 1), optional OUT_W_FULL / OUT_U_FULL (whole-model scope, 025-style)
  run config     the lambda grid and lr the main run reads (config_main.json, written from calibrate.py's
                 recommendation by an explicit ADOPT step; never auto-picked)
  matched size   per target: an OUT arm's value at the target's LAT circuit size, linear in log10(size) over the arm's
                 lambda grid (071 logic.interp_log_n); the nearest-size lambda for set overlaps
  overlap        Jaccard and both containments of two member sets
  bootstrap      paired over targets, B = 2000, fixed seed
  calibration    failure rate, monotonicity in lambda, size vs lambda, and the recommendation rule

  PYTHONPATH=src python -m pytest experiments/080-latent-vs-output/test_common.py -q
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

# ------------------------------------------------------------------------------------------- protocol constants
GAMMA = 0.25                 # gamma_C = gamma_A (protocol v1)
PROD_LAM = 1e-3              # the latent circuits' lambda (protocol v1)
PROD_LR = 0.05
PROD_STEPS = 400
PROD_WD = 0.05               # AdamW decay; lr x wd is held at PROD_LR x PROD_WD when lr changes (see weight_decay)
BAND_LO, BAND_HI, NEC_MIN = 0.8, 1.5, 0.9                     # the DAN-8 pass rule
LAT_FAITH = ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")    # latent Z, A_sp, C_sp (activation read)
LAT_NEC = "phi_sup_blind_tk"
OUT_FAITH = ("out_free0", "out_freeM_topk", "out_freeN_topk")  # output Z, A_sp, C_sp
OUT_NEC = "out_phi_sup_blind"
# The per-circuit metrics the report matches in size: short name -> row field
METRICS: Dict[str, str] = {"Z": "free0_tk", "A": "freeM_topk_tk", "C": "freeN_topk_tk", "nec": LAT_NEC,
                           "oZ": "out_free0", "oA": "out_freeM_topk", "oC": "out_freeN_topk", "onec": OUT_NEC}

# ------------------------------------------------------------------------------------------- tiers and bands
# Every target is an internal latent. Tiers say how much a target affects the next-token prediction; none of them
# means "not internal".
DP_HIGH_NATS = 1.0           # 064's direct-path tier cut: boost@peak >= 1 nat (fixed before 064's inference results)
DP_LOW_NATS = 0.1            # a tenth of it
TIERS: Tuple[str, ...] = ("dp_high", "dp_mid", "dp_low")      # legacy direct-path tiers, highest first
CAUSAL_TIERS: Tuple[str, ...] = ("high", "low")                # the causal tertiles (the middle one is dropped)
TIER_LABEL: Dict[str, str] = {"dp_high": "direct-path (064) high", "dp_mid": "direct-path (064) mid",
                              "dp_low": "direct-path (064) low", "high": "high next-token effect",
                              "low": "low next-token effect"}
# Rows written before the 2026-10-18 rename (the first smokes) carry the old names
LEGACY_TIER_ALIAS: Dict[str, str] = {"writer": "dp_high", "weak": "dp_mid", "internal": "dp_low"}
BANDS: Tuple[str, ...] = ("L0-4", "L5-7", "L8-11")
KINDS: Tuple[str, ...] = ("attn", "mlp", "resid")
N_PER_TIER = 50
N_CALIB = 12
TARGET_SEED = 20261018


def tier_of(boost_peak: float) -> str:
    """Legacy direct-path tier from 064's boost_peak (nats the target's top promoted token gains over the average
    token at the target's peak activation, direct path through the final norm, context-free)."""
    if boost_peak >= DP_HIGH_NATS:
        return "dp_high"
    if boost_peak < DP_LOW_NATS:
        return "dp_low"
    return "dp_mid"


def norm_tier(t: Optional[str]) -> Optional[str]:
    """A tier name in the current vocabulary (old rows' writer / weak / internal mapped to dp_*)."""
    return LEGACY_TIER_ALIAS.get(t, t) if t else t


def band(layer: int) -> str:
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def parse(key: str) -> Tuple[int, str, int]:
    l, k, i = key.split(".")
    return int(l), k, int(i)


def cell_quotas(n_per_tier: int = N_PER_TIER, bands: Sequence[str] = BANDS) -> Dict[str, int]:
    """Targets per band within a tier: an equal share, the remainder to the deepest bands (50 -> 16 / 17 / 17)."""
    base, rem = divmod(int(n_per_tier), len(bands))
    return {b: base + (1 if i >= len(bands) - rem else 0) for i, b in enumerate(bands)}


def stratified_pick(pool: Mapping[Tuple[str, str], Sequence[str]], n_per_tier: int = N_PER_TIER,
                    seed: int = TARGET_SEED, exclude: Iterable[str] = ()) -> List[str]:
    """The main target list: per tier, cell_quotas over the bands, drawn uniformly (fixed seed, keys sorted first) from
    pool[(tier, band)]. A cell short of its quota gives the shortfall to the tier's other bands (deepest first).
    Cells interleaved (tier-major round robin) so a run stopped part-way is still stratified."""
    rng = random.Random(seed)
    excl = set(exclude)
    picks: Dict[Tuple[str, str], List[str]] = {}
    for t in TIERS:
        quota = cell_quotas(n_per_tier)
        avail = {b: sorted(k for k in pool.get((t, b), ()) if k not in excl) for b in BANDS}
        short = 0
        for b in BANDS:
            if len(avail[b]) < quota[b]:
                short += quota[b] - len(avail[b])
                quota[b] = len(avail[b])
        for b in reversed(BANDS):                     # hand the shortfall to bands with room, deepest first
            give = min(short, len(avail[b]) - quota[b])
            quota[b] += give
            short -= give
        for b in BANDS:
            picks[(t, b)] = rng.sample(avail[b], quota[b])
    cells = [(t, b) for b in BANDS for t in TIERS]
    depth = max((len(v) for v in picks.values()), default=0)
    return [picks[c][i] for i in range(depth) for c in cells if i < len(picks[c])]


# ------------------------------------------------------------------------------------------- within-kind sampling
# withinkind (the default between the two 2026-10-18 reviews): under tier x band sampling, the direct-path tier is
# confounded with site kind (dp_low targets were 48 of 54 attention, dp_high 51 of 54 residual; the population leaves
# no other way to fill those cells). So that sample is a set of WITHIN-KIND, WITHIN-BAND contrasts between ADJACENT
# direct-path tiers, layer-matched within each pair. SAMPLING=causal (now the default) applies the same within-group,
# layer-matched pairing to the causal tiers; tierband stays reproducible (stratified_pick / calib_pick).
SAMPLINGS: Tuple[str, ...] = ("causal", "withinkind", "tierband")
PAIRS: Tuple[Tuple[str, str, str, str], ...] = (       # withinkind: (kind, band, higher tier, lower tier)
    ("resid", "L5-7", "dp_high", "dp_mid"),
    ("resid", "L8-11", "dp_high", "dp_mid"),
    ("mlp", "L8-11", "dp_high", "dp_mid"),
    ("mlp", "L0-4", "dp_mid", "dp_low"),
    ("attn", "L0-4", "dp_mid", "dp_low"),
    ("attn", "L8-11", "dp_mid", "dp_low"),
)
# causal: one high-vs-low pair per kind x band group
CAUSAL_PAIRS: Tuple[Tuple[str, str, str, str], ...] = tuple((k, b, "high", "low") for k in KINDS for b in BANDS)


def design_pairs(tiers_present: Iterable[str]) -> Tuple[Tuple[str, str, str, str], ...]:
    """The pairs P1 contrasts: the causal design's 9 if any causal tier is present, else withinkind's 6."""
    return CAUSAL_PAIRS if set(tiers_present) & set(CAUSAL_TIERS) else PAIRS
N_PER_CELL = 13                                         # 6 pairs x 2 tiers x 13 = 156
SMOKE_TARGETS: Tuple[str, ...] = ("1.mlp.35042", "11.resid.15522")   # kept from the first smoke (both in pair cells)


def band_layers(b: str) -> List[int]:
    return {"L0-4": [0, 1, 2, 3, 4], "L5-7": [5, 6, 7], "L8-11": [8, 9, 10, 11]}[b]


def pair_name(kind: str, b: str) -> str:
    return "%s %s" % (kind, b)


def cell_name(kind: str, b: str, tier: str) -> str:
    return "%s %s %s" % (kind, b, tier)


def pair_of(kind: str, b: str, tier: str, pairs=None) -> Optional[Tuple[str, str, str, str]]:
    """The pair a (kind, band, tier) cell belongs to (in `pairs`; default: the design its tier implies), or None."""
    if pairs is None:
        pairs = CAUSAL_PAIRS if tier in CAUSAL_TIERS else PAIRS
    for p in pairs:
        if p[0] == kind and p[1] == b and tier in (p[2], p[3]):
            return p
    return None


def layer_matched_quota(avail_hi: Mapping[int, int], avail_lo: Mapping[int, int], n: int,
                        rng: random.Random) -> Dict[int, int]:
    """Per-layer counts for BOTH tiers of a pair, so the two tiers have the same layer histogram: a layer can take at
    most min(available in either tier); quotas are dealt one at a time round robin over the layers in one seeded
    order (spreading the sample across the band) until n or the capacity runs out."""
    layers = sorted(set(avail_hi) | set(avail_lo))
    cap = {l: min(int(avail_hi.get(l, 0)), int(avail_lo.get(l, 0))) for l in layers}
    order = layers[:]
    rng.shuffle(order)
    quota = {l: 0 for l in layers}
    total = 0
    while total < n and any(quota[l] < cap[l] for l in layers):
        for l in order:
            if total < n and quota[l] < cap[l]:
                quota[l] += 1
                total += 1
    return quota


def withinkind_pick(pool: Mapping[Tuple[str, str, int], Sequence[str]], n_per_cell: int = N_PER_CELL,
                    seed: int = TARGET_SEED, exclude: Iterable[str] = (), pairs=PAIRS) -> Tuple[List[str], Dict]:
    """The main list under SAMPLING=withinkind (pairs=PAIRS) and, with pairs=CAUSAL_PAIRS and a pool of causal tiers,
    under SAMPLING=causal. pool[(kind, tier, layer)] = keys. Per pair: layer-matched
    quotas (layer_matched_quota), each tier drawn uniformly per layer (fixed seed, keys sorted first). If the matched
    capacity is short of n_per_cell, each tier is topped up from its remaining keys anywhere in the band (flagged
    unmatched). Cells interleaved (pair-major round robin, higher tier first) so a run stopped part-way is still
    balanced. Returns (keys, info): info[pair name] = quota, matched count, per-tier picks and unmatched keys."""
    rng = random.Random(seed)
    excl = set(exclude)
    picks: Dict[Tuple[str, str, str], List[str]] = {}
    info: Dict[str, Dict] = {}
    for kind, b, hi, lo in pairs:
        layers = band_layers(b)
        av = {t: {l: sorted(k for k in pool.get((kind, t, l), ()) if k not in excl) for l in layers} for t in (hi, lo)}
        quota = layer_matched_quota({l: len(av[hi][l]) for l in layers}, {l: len(av[lo][l]) for l in layers},
                                    n_per_cell, rng)
        unmatched: Dict[str, List[str]] = {}
        for t in (hi, lo):
            chosen = []
            for l in layers:
                chosen += rng.sample(av[t][l], quota[l])
            short = n_per_cell - len(chosen)
            extra: List[str] = []
            if short > 0:
                rest = sorted(k for l in layers for k in av[t][l] if k not in set(chosen))
                extra = rng.sample(rest, min(short, len(rest)))
            picks[(kind, b, t)] = chosen + extra
            unmatched[t] = extra
        n_av = {t: sum(len(v) for v in av[t].values()) for t in (hi, lo)}
        info[pair_name(kind, b)] = dict(quota=quota, n_matched=sum(quota.values()), hi=hi, lo=lo,
                                        picks={t: picks[(kind, b, t)] for t in (hi, lo)}, unmatched=unmatched,
                                        n_available=n_av, thin=any(len(picks[(kind, b, t)]) < n_per_cell
                                                                   for t in (hi, lo)))
    cells = [(k, b, t) for k, b, hi, lo in pairs for t in (hi, lo)]
    depth = max((len(v) for v in picks.values()), default=0)
    return [picks[c][i] for i in range(depth) for c in cells if i < len(picks[c])], info


def withinkind_calib(pool: Mapping[Tuple[str, str, int], Sequence[str]], exclude: Iterable[str] = (),
                     force: Iterable[str] = SMOKE_TARGETS, seed: int = TARGET_SEED + 1, pairs=PAIRS) -> List[str]:
    """The calibration list: one target per cell of `pairs` (withinkind 12, causal 18), disjoint from `exclude`,
    layer-matched within each pair. A pair's anchor is a `force` key in one of its cells (the smoke targets, so smoke
    stays a subset of calibration wherever it sits in a cell) or else a random key of the higher tier; the partner is
    drawn from the other tier at the anchor's layer, or the nearest layer that has one."""
    rng = random.Random(seed)
    excl = set(exclude)
    force = list(force)
    out: List[str] = []
    for kind, b, hi, lo in pairs:
        layers = band_layers(b)
        av = {t: {l: sorted(k for k in pool.get((kind, t, l), ()) if k not in excl) for l in layers} for t in (hi, lo)}
        anchor = None
        for t in (hi, lo):
            for l in layers:
                for k in force:
                    if k in pool.get((kind, t, l), ()):
                        anchor = (t, l, k)
        if anchor is None:
            cand = sorted(k for l in layers for k in av[hi][l])
            if not cand:
                continue
            k = rng.choice(cand)
            anchor = (hi, parse(k)[0], k)
        t0, l0, k0 = anchor
        other = lo if t0 == hi else hi
        partner = None
        for l in sorted(layers, key=lambda x: (abs(x - l0), x)):
            if av[other][l]:
                partner = rng.choice(av[other][l])
                break
        out += [k0] + ([partner] if partner else [])
    return out


# ------------------------------------------------------------------------------------------- causal sampling
N_PER_CAUSAL = 9                                        # per tier per group: 9 groups x 2 x 9 = 162
EFFECT_DIR = HERE / "effect"
EFFECTS_FILE = EFFECT_DIR / "effects.parquet"
EFFECT_MEASURE = "effect_abs"                           # the sampling measure (next_token_effect.py explains why)
MIN_EFFECT_CONTEXTS = 8                                 # fewer measured contexts: the target is left out of the pool


def load_effects(path: Path = EFFECTS_FILE) -> Dict[str, Dict]:
    """{key: effect row} from next_token_effect.py's parquet (rows with an error or fewer than MIN_EFFECT_CONTEXTS
    contexts dropped). Refuses a missing file: SAMPLING=causal has nothing to sample from until the measurement ran."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "%s does not exist. SAMPLING=causal needs the in-context next-token effect of every pool target: run the "
            "measurement first (experiments/080-latent-vs-output/queue_effect.sh, which runs next_token_effect.py and "
            "writes this file), or use SAMPLING=withinkind / tierband." % path)
    import pandas as pd
    d = pd.read_parquet(path)
    out = {}
    for r in d.to_dict("records"):
        if r.get("error") and isinstance(r.get("error"), str):
            continue
        if num(r.get(EFFECT_MEASURE)) is None or int(r.get("n_contexts") or 0) < MIN_EFFECT_CONTEXTS:
            continue
        out[r["seed"]] = r
    return out


def causal_tiers(effects: Mapping[str, float]) -> Dict[str, str]:
    """Tiers within ONE kind x band group's measured pool: rank the targets by effect (ties broken by key, so the split
    is deterministic); the top floor(n / 3) are "high", the bottom floor(n / 3) "low"; the middle third is dropped."""
    ranked = sorted(effects, key=lambda k: (float(effects[k]), k))
    n3 = len(ranked) // 3
    out = {k: "low" for k in ranked[:n3]}
    out.update({k: "high" for k in ranked[len(ranked) - n3:]} if n3 else {})
    return out


def causal_pool(meta: Mapping[str, Mapping]) -> Tuple[Dict[Tuple[str, str, int], List[str]], Dict[str, str], Dict]:
    """From {key: dict(kind, band, layer, effect_abs)} (the measured, filtered pool): pool[(kind, tier, layer)] for
    withinkind_pick(pairs=CAUSAL_PAIRS), the tier of every key that has one, and per group the pool size and the
    tertile cut-offs (nats)."""
    groups: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for k, m in meta.items():
        groups[(m["kind"], m["band"])][k] = float(m[EFFECT_MEASURE])
    pool: Dict[Tuple[str, str, int], List[str]] = defaultdict(list)
    tiers: Dict[str, str] = {}
    cuts: Dict[str, Dict] = {}
    for (kind, b), eff in sorted(groups.items()):
        t = causal_tiers(eff)
        tiers.update(t)
        for k, tier in t.items():
            pool[(kind, tier, int(meta[k]["layer"]))].append(k)
        lows = [eff[k] for k, v in t.items() if v == "low"]
        highs = [eff[k] for k, v in t.items() if v == "high"]
        cuts[pair_name(kind, b)] = dict(n_pool=len(eff), n_low=len(lows), n_high=len(highs),
                                        low_max=max(lows) if lows else None, high_min=min(highs) if highs else None)
    return dict(pool), tiers, cuts


def load_meta(here: Path = HERE) -> Dict[str, Dict]:
    """Per-target metadata (tier, band, boost_peak, lat_n, cell, ...) from every targets_meta*.csv here, the default
    sampling's targets_meta.csv read last (tier / band / boost_peak are properties of the target, so the files agree
    wherever they overlap)."""
    import csv
    files = sorted(Path(here).glob("targets_meta*.csv"), key=lambda p: (p.name == "targets_meta.csv", p.name))
    meta: Dict[str, Dict] = {}
    for p in files:
        for r in csv.DictReader(open(p)):
            meta[r["seed"]] = dict(meta.get(r["seed"], {}), **r)
    return meta


def calib_pick(pool: Mapping[Tuple[str, str], Sequence[str]], n: int = N_CALIB, seed: int = TARGET_SEED + 1,
               exclude: Iterable[str] = ()) -> List[str]:
    """The calibration targets, disjoint from the main list: one per (tier, band) cell, then the rest from the
    deepest band, one per tier in turn (12 = 9 + 3 x L8-11)."""
    rng = random.Random(seed)
    excl = set(exclude)
    avail = {c: sorted(k for k in pool.get(c, ()) if k not in excl) for c in [(t, b) for t in TIERS for b in BANDS]}
    out: List[str] = []
    for b in BANDS:
        for t in TIERS:
            if avail[(t, b)] and len(out) < n:
                k = rng.choice(avail[(t, b)])
                avail[(t, b)].remove(k)
                out.append(k)
    i = 0
    while len(out) < n and any(avail[(t, BANDS[-1])] for t in TIERS):
        t = TIERS[i % len(TIERS)]
        if avail[(t, BANDS[-1])]:
            k = rng.choice(avail[(t, BANDS[-1])])
            avail[(t, BANDS[-1])].remove(k)
            out.append(k)
        i += 1
    return out


# ------------------------------------------------------------------------------------------- arms
ARMS: Dict[str, Dict[str, object]] = {
    "LAT": dict(endpoint="latent", weighted=True, scope="upstream", fit=False),
    "OUT_W": dict(endpoint="output", weighted=True, scope="upstream", fit=True),
    "OUT_U": dict(endpoint="output", weighted=False, scope="upstream", fit=True),
    "OUT_W_FULL": dict(endpoint="output", weighted=True, scope="all", fit=True),      # optional, off by default
    # the unweighted pair of OUT_W_FULL: with every site maskable no layer after the circuit runs live to rescale
    # its members, so this pair measures the coefficients' effect on the output endpoint without that help
    "OUT_U_FULL": dict(endpoint="output", weighted=False, scope="all", fit=True),     # optional, off by default
}
DEFAULT_ARMS: Tuple[str, ...] = ("LAT", "OUT_W", "OUT_U")
FIT_ARMS: Tuple[str, ...] = ("OUT_W", "OUT_U", "OUT_W_FULL", "OUT_U_FULL")


def lam_tag(lam: float) -> str:
    return "%g" % lam


def weight_decay(lr: float) -> float:
    """AdamW decay for a given lr: lr x wd held at production's 0.05 x 0.05, so the decay per step (and
    steps x lr x wd = 1.0 at 400 steps, the config's calibrated product) is unchanged when calibration moves lr."""
    return PROD_LR * PROD_WD / float(lr)


# ------------------------------------------------------------------------------------------- run config
CONFIG_KEYS = ("arms", "logit_norm", "steps")


def load_run_config(path: Path) -> dict:
    """The main run's config: {"arms": {arm: {"lams": [...], "lr": x}}, "logit_norm", "steps", ...}. Refuses a
    missing file (no silent default) and an arm without a lambda grid or lr."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "%s does not exist: run calibrate.py, read calib/recommendation.json, then adopt it with ADOPT=1 "
            "calibrate.py (or write the file by hand). The main run never picks its lambda grid itself." % path)
    cfg = json.loads(path.read_text())
    for k in CONFIG_KEYS:
        if k not in cfg:
            raise ValueError("%s: missing %r" % (path, k))
    for arm, a in cfg["arms"].items():
        if arm not in FIT_ARMS:
            raise ValueError("%s: unknown fitted arm %r" % (path, arm))
        if not a.get("lams") or not a.get("lr"):
            raise ValueError("%s: arm %s needs a non-empty 'lams' and an 'lr'" % (path, arm))
    return cfg


# ------------------------------------------------------------------------------------------- numbers
def num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def passes_latent(row: Mapping) -> bool:
    """DAN-8 on the latent endpoint (activation read): Z, A, C in [0.8, 1.5] and necessity >= 0.9."""
    vals = [num(row.get(k)) for k in LAT_FAITH]
    nec = num(row.get(LAT_NEC))
    return all(v is not None and BAND_LO <= v <= BAND_HI for v in vals) and nec is not None and nec >= NEC_MIN


def passes_output(row: Mapping) -> bool:
    """The same rule on the output endpoint (eval.wcm_eval.passes_output)."""
    vals = [num(row.get(k)) for k in OUT_FAITH]
    nec = num(row.get(OUT_NEC))
    return all(v is not None and BAND_LO <= v <= BAND_HI for v in vals) and nec is not None and nec >= NEC_MIN


# ------------------------------------------------------------------------------------------- matched size
def _load_071_logic():
    """071's logic module, loaded by path under its own name (putting 071 on sys.path would let its report.py /
    sweep.py shadow this experiment's)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_logic_071", EXP / "071-contrast-sweep" / "logic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L071 = _load_071_logic()
# y at circuit size n_star, linear in log10(n) through one arm's (size, y) points over its lambda grid; outside the
# range the nearest endpoint, flagged. Returns (y, extrapolated). 071's own function, reused unchanged.
interp_log_n = L071.interp_log_n


def nearest_in_log(ns: Sequence, n_star: float) -> Optional[int]:
    """Index of the size nearest n_star in log10 (ties: the smaller size); None when nothing is usable."""
    best, bi = None, None
    for i, n in enumerate(ns):
        n = num(n)
        if n is None or n <= 0 or num(n_star) is None or n_star <= 0:
            continue
        d = abs(math.log10(n) - math.log10(n_star))
        if best is None or d < best - 1e-12 or (abs(d - best) <= 1e-12 and n < num(ns[bi])):
            best, bi = d, i
    return bi


def size_to_reach(ns: Sequence, ys: Sequence, thr: float) -> Tuple[Optional[float], str]:
    """The smallest circuit size at which y first reaches thr, walking the (size, y) points from small to large and
    interpolating linearly in log10(size) at the first crossing. Returns (size, status): status "ok", "already"
    (the smallest circuit already reaches it: an upper bound), or "never" (size None)."""
    pts = sorted((num(n), num(y)) for n, y in zip(ns, ys) if num(n) and num(n) > 0 and num(y) is not None)
    if not pts:
        return None, "never"
    if pts[0][1] >= thr:
        return pts[0][0], "already"
    for (n0, y0), (n1, y1) in zip(pts, pts[1:]):
        if y1 >= thr:
            t = (thr - y0) / (y1 - y0) if y1 != y0 else 1.0
            return 10 ** (math.log10(n0) + t * (math.log10(n1) - math.log10(n0))), "ok"
    return None, "never"


# ------------------------------------------------------------------------------------------- overlap
def overlap(a: Iterable, b: Iterable) -> Dict[str, Optional[float]]:
    """Jaccard |a & b| / |a | b|, containment of a in b |a & b| / |a|, of b in a |a & b| / |b|."""
    a, b = set(a), set(b)
    i, u = len(a & b), len(a | b)
    return dict(jaccard=(i / u) if u else None, cont_a_in_b=(i / len(a)) if a else None,
                cont_b_in_a=(i / len(b)) if b else None, n_inter=i)


# ------------------------------------------------------------------------------------------- bootstrap
B_BOOT = 2000
BOOT_SEED = 20261018


def _stat(vals: Sequence[float], stat: str) -> float:
    import numpy as np
    v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return float("nan")
    return float(np.mean(v) if stat == "mean" else np.median(v))


def paired_bootstrap(a: Sequence, b: Optional[Sequence] = None, stat: str = "median", B: int = B_BOOT,
                     seed: int = BOOT_SEED) -> Dict[str, float]:
    """Paired bootstrap over targets (one entry per target in a and b, same order; None/NaN = missing for that arm).
    Returns the statistic (median or mean) of a, of b, and of the paired differences a - b (targets where both
    exist), each with a 95% percentile interval from B resamples of targets."""
    import numpy as np
    a = np.array([np.nan if num(x) is None else float(x) for x in a], dtype=float)
    out: Dict[str, float] = {"n": int(np.isfinite(a).sum())}
    f = np.nanmean if stat == "mean" else np.nanmedian
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(B, n)) if n else np.zeros((B, 0), dtype=int)

    def ci(x):
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pt = float(f(x)) if np.isfinite(x).any() else float("nan")
                bs = np.array([f(x[i]) if np.isfinite(x[i]).any() else np.nan for i in idx])
        bs = bs[np.isfinite(bs)]
        lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs.size else (np.nan, np.nan))
        return pt, float(lo), float(hi)

    out["a"], out["a_lo"], out["a_hi"] = ci(a)
    if b is not None:
        b = np.array([np.nan if num(x) is None else float(x) for x in b], dtype=float)
        assert len(b) == n, "paired bootstrap needs one entry per target in both arms"
        out["n_b"] = int(np.isfinite(b).sum())
        out["b"], out["b_lo"], out["b_hi"] = ci(b)
        d = np.where(np.isfinite(a) & np.isfinite(b), a - b, np.nan)
        out["n_pair"] = int(np.isfinite(d).sum())
        out["d"], out["d_lo"], out["d_hi"] = ci(d)
    return out


def spearman_bootstrap(x: Sequence, y: Sequence, B: int = B_BOOT, seed: int = BOOT_SEED) -> Dict[str, float]:
    """Spearman correlation of x and y over targets with both values, with a 95% bootstrap interval."""
    import numpy as np
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if num(a) is not None and num(b) is not None]
    if len(pairs) < 3:
        return dict(rho=float("nan"), lo=float("nan"), hi=float("nan"), n=len(pairs))
    xa, ya = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])

    def rho(i):
        rx = np.argsort(np.argsort(xa[i])).astype(float)
        ry = np.argsort(np.argsort(ya[i])).astype(float)
        if rx.std() == 0 or ry.std() == 0:
            return np.nan
        return float(np.corrcoef(rx, ry)[0, 1])

    rng = np.random.default_rng(seed)
    full = rho(np.arange(len(xa)))
    bs = np.array([rho(rng.integers(0, len(xa), len(xa))) for _ in range(B)])
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs.size else (np.nan, np.nan)
    return dict(rho=full, lo=float(lo), hi=float(hi), n=len(pairs))


def pair_contrast(cells: Mapping[str, Tuple[Sequence, Sequence]], B: int = B_BOOT,
                  seed: int = BOOT_SEED) -> Dict:
    """P1's primary test. cells[pair] = (values of the higher tier's targets, values of the lower tier's). Per pair
    d = median(higher) - median(lower); pooled = the SIMPLE MEAN of the pair differences (every kind x band contrast
    weighted equally; no dependence on noisy per-cell variance estimates). Bootstrap: targets resampled within each of
    the cells, independently, B times; percentile 95% intervals for every d and for the pooled mean. Also reported, as
    a sensitivity: the inverse-variance pooled difference (weights 1 / bootstrap variance of each d, held fixed across
    resamples). Pairs with an empty tier are left out (counted in n_pairs)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    clean = {}
    for p, (hi, lo) in cells.items():
        h = np.array([float(v) for v in hi if num(v) is not None], dtype=float)
        l = np.array([float(v) for v in lo if num(v) is not None], dtype=float)
        if len(h) and len(l):
            clean[p] = (h, l)
    out = dict(per_pair={}, n_pairs=len(clean))
    if not clean:
        out.update(pooled=float("nan"), pooled_lo=float("nan"), pooled_hi=float("nan"), iv=float("nan"),
                   iv_lo=float("nan"), iv_hi=float("nan"))
        return out
    names = list(clean)
    pt = np.array([np.median(clean[p][0]) - np.median(clean[p][1]) for p in names])
    bs = np.empty((B, len(names)))
    for b in range(B):
        for j, p in enumerate(names):
            h, l = clean[p]
            bs[b, j] = np.median(h[rng.integers(0, len(h), len(h))]) - np.median(l[rng.integers(0, len(l), len(l))])
    var = bs.var(axis=0)
    w = np.where(var > 0, 1.0 / np.where(var > 0, var, 1.0), 0.0)
    if w.sum() == 0:
        w = np.ones_like(w)
    for j, p in enumerate(names):
        lo_, hi_ = np.percentile(bs[:, j], [2.5, 97.5])
        out["per_pair"][p] = dict(d=float(pt[j]), lo=float(lo_), hi=float(hi_), n_hi=len(clean[p][0]),
                                  n_lo=len(clean[p][1]))
    pooled_bs = bs.mean(axis=1)
    iv_bs = bs @ w / w.sum()
    out.update(pooled=float(pt.mean()), pooled_lo=float(np.percentile(pooled_bs, 2.5)),
               pooled_hi=float(np.percentile(pooled_bs, 97.5)), iv=float(pt @ w / w.sum()),
               iv_lo=float(np.percentile(iv_bs, 2.5)), iv_hi=float(np.percentile(iv_bs, 97.5)))
    return out


def fe_slope(x: Sequence, y: Sequence, groups: Sequence[str], B: int = B_BOOT, seed: int = BOOT_SEED) -> Dict:
    """P1's secondary test: OLS of y on log(x) with one fixed effect (intercept) per group (kind x band), so the slope
    is estimated from within-group variation in output-facingness only. Bootstrap: targets resampled within each
    group, B times; percentile 95% interval on the slope. Targets with x <= 0 or a missing y are dropped; groups with
    fewer than 2 targets carry no within-group information and are dropped."""
    import numpy as np
    rows = [(math.log(float(a)), float(b), g) for a, b, g in zip(x, y, groups)
            if num(a) is not None and float(a) > 0 and num(b) is not None]
    by = defaultdict(list)
    for r in rows:
        by[r[2]].append(r)
    by = {g: v for g, v in by.items() if len(v) >= 2}
    gs = sorted(by)
    if not gs:
        return dict(slope=float("nan"), lo=float("nan"), hi=float("nan"), n=0, n_groups=0)

    def fit(sample):
        X = np.array([[r[0]] + [1.0 if r[2] == g else 0.0 for g in gs] for r in sample])
        Y = np.array([r[1] for r in sample])
        if np.linalg.matrix_rank(X) < X.shape[1]:
            return float("nan")
        return float(np.linalg.lstsq(X, Y, rcond=None)[0][0])

    full = fit([r for g in gs for r in by[g]])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        s = [by[g][i] for g in gs for i in rng.integers(0, len(by[g]), len(by[g]))]
        bs.append(fit(s))
    bs = np.array(bs)
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs.size else (float("nan"), float("nan"))
    return dict(slope=full, lo=float(lo), hi=float(hi), n=sum(len(v) for v in by.values()), n_groups=len(gs))


# ------------------------------------------------------------------------------------------- calibration
FAIL_FAITH = 0.05           # a fit "fails" when its held-out output Z faithfulness is missing or below this
MONO_TOL = 0.02              # faithfulness may rise by this much with lambda and still count as monotone
GRID_LEN = 4                 # lambda values the main run fits per arm


def fit_failed(row: Mapping) -> bool:
    """Calibration failure: the fit errored or was rejected, or its held-out output Z faithfulness is missing,
    negative or ~0 (< FAIL_FAITH)."""
    if row.get("error") or row.get("skip"):
        return True
    v = num(row.get("out_free0"))
    return v is None or v < FAIL_FAITH


def monotone(lams: Sequence[float], vals: Sequence, increasing: bool = False, tol: float = 0.0) -> Optional[bool]:
    """Whether vals is monotone in lambda (non-increasing by default) within tol; None with fewer than 2 values."""
    pts = sorted((float(l), num(v)) for l, v in zip(lams, vals) if num(v) is not None)
    if len(pts) < 2:
        return None
    sgn = 1.0 if increasing else -1.0
    return all(sgn * (b[1] - a[1]) >= -tol for a, b in zip(pts, pts[1:]))


def calib_summary(rows: Sequence[Mapping], lat_n: Mapping[str, float]) -> Dict:
    """Per (arm, lr): failure rate over all fits, per-target monotonicity of size and of output Z in lambda, the
    median size and median output Z per lambda, and the LAT size quartiles of the calibration targets."""
    import numpy as np
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], float(r["lr"]))].append(r)
    out = {}
    for (arm, lr), rs in sorted(by.items()):
        per_t = defaultdict(list)
        for r in rs:
            per_t[r["seed"]].append(r)
        mono_n, mono_f = [], []
        for t, trs in per_t.items():
            ok = [r for r in trs if not (r.get("error") or r.get("skip"))]
            lams = [float(r["lam"]) for r in ok]
            mono_n.append(monotone(lams, [r.get("n") for r in ok]))
            mono_f.append(monotone(lams, [r.get("out_free0") for r in ok], tol=MONO_TOL))
        lam_stats = {}
        for lam in sorted({float(r["lam"]) for r in rs}):
            lr_ = [r for r in rs if float(r["lam"]) == lam]
            ns = [num(r.get("n")) for r in lr_ if num(r.get("n")) is not None]
            fz = [num(r.get("out_free0")) for r in lr_ if num(r.get("out_free0")) is not None]
            lam_stats[lam_tag(lam)] = dict(
                n_fits=len(lr_), fail=float(np.mean([fit_failed(r) for r in lr_])),
                n_median=float(np.median(ns)) if ns else None, oZ_median=float(np.median(fz)) if fz else None,
                oZ_neg=int(sum(v < 0 for v in fz)))
        frac = lambda xs: (float(np.mean([bool(x) for x in xs if x is not None]))     # noqa: E731
                           if any(x is not None for x in xs) else None)
        out["%s|%g" % (arm, lr)] = dict(arm=arm, lr=lr, n_fits=len(rs), n_targets=len(per_t),
                                        fail_rate=float(np.mean([fit_failed(r) for r in rs])),
                                        mono_size=frac(mono_n), mono_faith=frac(mono_f), lams=lam_stats)
    ln = [float(v) for v in lat_n.values() if num(v)]
    out["_lat"] = dict(n=len(ln), p10=float(np.percentile(ln, 10)) if ln else None,
                       p25=float(np.percentile(ln, 25)) if ln else None,
                       median=float(np.median(ln)) if ln else None,
                       p75=float(np.percentile(ln, 75)) if ln else None,
                       p90=float(np.percentile(ln, 90)) if ln else None)
    return out


def choose_grid(lam_medians: Mapping[float, Optional[float]], lat_lo: float, lat_hi: float,
                k: int = GRID_LEN) -> Tuple[List[float], bool]:
    """k consecutive swept lambdas whose median circuit sizes best cover [lat_lo, lat_hi] (the calibration targets'
    LAT size p10..p90) in log10 size: the window maximising the covered fraction of that interval, ties broken by
    the smaller gap between the window's log-size midpoint and the interval's. Returns (lambdas ascending,
    brackets): brackets = the window's sizes span the whole interval."""
    pts = sorted((float(l), num(n)) for l, n in lam_medians.items() if num(n) and num(n) > 0)
    if not pts:
        return [], False
    k = min(k, len(pts))
    lo, hi = math.log10(lat_lo), math.log10(lat_hi)
    best = None
    for s in range(len(pts) - k + 1):
        w = pts[s:s + k]
        a, b = sorted(math.log10(n) for _, n in w)[0], sorted(math.log10(n) for _, n in w)[-1]
        cov = max(0.0, min(b, hi) - max(a, lo)) / max(hi - lo, 1e-9)
        mid = abs((a + b) / 2 - (lo + hi) / 2)
        key = (-cov, mid)
        if best is None or key < best[0]:
            best = (key, [l for l, _ in w], a <= lo and b >= hi)
    return sorted(best[1]), best[2]


def recommend(summary: Mapping, prod_lr: float = PROD_LR, k: int = GRID_LEN) -> Dict:
    """The calibration recommendation per fitted arm: the lr with the lowest failure rate (ties: the higher
    faithfulness-monotone fraction, then the lr nearest production's 0.05 in log), and at that lr the k-lambda window
    of choose_grid over the LAT size p10..p90. Warnings when nothing brackets the LAT sizes or failures stay high."""
    lat = summary["_lat"]
    rec = {"arms": {}, "warnings": []}
    arms = sorted({v["arm"] for kk, v in summary.items() if not kk.startswith("_")})
    for arm in arms:
        cands = [v for kk, v in summary.items() if not kk.startswith("_") and v["arm"] == arm]
        cands.sort(key=lambda v: (v["fail_rate"], -(v["mono_faith"] or 0.0), abs(math.log10(v["lr"] / prod_lr))))
        best = cands[0]
        meds = {float(l): s["n_median"] for l, s in best["lams"].items()}
        grid, brackets = choose_grid(meds, lat["p10"], lat["p90"], k) if lat["p10"] else ([], False)
        rec["arms"][arm] = dict(lr=best["lr"], lams=grid, fail_rate=best["fail_rate"],
                                mono_faith=best["mono_faith"], mono_size=best["mono_size"], brackets_lat=brackets)
        if not brackets:
            rec["warnings"].append("%s: no %d-lambda window of the swept grid brackets the LAT sizes "
                                   "(p10 %.0f .. p90 %.0f); extend the sweep before adopting" % (arm, k, lat["p10"] or 0,
                                                                                               lat["p90"] or 0))
        if best["fail_rate"] > 0.25:
            rec["warnings"].append("%s: failure rate %.0f%% at the best lr; the logit objective is not stable here"
                                   % (arm, 100 * best["fail_rate"]))
    return rec
