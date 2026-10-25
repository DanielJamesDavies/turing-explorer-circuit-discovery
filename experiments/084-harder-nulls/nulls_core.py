"""DAN-181: the pure parts of the harder random-circuit nulls (no model, no driver import), so the tests can import
them: target order, depth bands, anchor lags, decile binning, the matched samplers, coefficient statistics and the
per-set profile summaries. harder_nulls.py runs them against the model; report.py reads the rows they produce.

Definitions (README "Matching definitions" states them for the paper):
  anchor lag        for a latent at an upstream site on one training context with anchor a (the target's argmax
                    position), lag = a - p for every position p <= a where its post-Top-K activation is > 0; a latent's
                    lag is its minimum over the fit's training contexts (positions after the anchor are ignored)
  anchor-active     lag <= WIN (default 8, decided 2026-10-24: active at the anchor or at one of the 8 positions before
                    it, on at least one training context; WIN = 3 left 16-34% of the production members out on 3 of 9
                    profiled targets)
  pool              the anchor-active latents of each production site, production members INCLUDED (decided
                    2026-10-24: a plain draw from the matched pool, the standard random baseline); the overlap with the
                    production members is recorded per draw
  frequency decile  per site, over the latents with a nonzero corpus activation count (latent_stats.active_count):
                    floor(10 * rank / n_alive), rank ascending (ties by latent index); a never-active latent gets -1
  contribution      per latent, the mean over the training contexts of |sum over p <= anchor of a_p * d pre / d a_p|,
                    pre = the target's encoder pre-activation at the anchor, a = the latent's natural code (activation
                    x gradient, gradients carried through every substituted upstream site)
  contribution bin  (what the contrib null matches on) per target, the production members' own contribution deciles:
                    edges at their 10th..90th percentiles, bins 0..9, -1 below their minimum, 10 above their maximum
  contribution dec. (reported) per target, rank deciles over the union of every site's anchor-active pool and the
                    production members
"""
import math
import random
import zlib
from collections import defaultdict

import numpy as np

LAG_NONE = 10 ** 6                     # "never active in the causal prefix of any training context"
WIN_DEFAULT = 8                        # the anchor-active window (decided 2026-10-24; was 3)
PANEL_SEED = 20261024                  # the TuringLLM panel's draw (make_panel.py)
COVER_WINDOWS = (0, 1, 2, 3, 4, 8, 16, 63)
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]     # Z, A, C (sparsity-preserving)
NEC = "phi_sup_blind_tk"
IND = "phi_cf_alpha_blind_tk"
BAND = (0.8, 1.5)
NEC_MIN = 0.9
TRAIN_EVAL_N = 16                      # training-split score: 16 of the fit's strongest training contexts


# ----------------------------------------------------------------------------------------------- targets
def band(layer, n_layer=12):
    """Depth band. 12-layer models (TuringLLM, GPT-2 small) use the paper's L0-4 / L5-7 / L8-11; other depths are cut
    into thirds (Pythia-70m: L0-1 / L2-3 / L4-5, pod plan D7)."""
    layer, n_layer = int(layer), int(n_layer)
    if n_layer == 12:
        return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")
    cuts = [int(round(n_layer * j / 3.0)) for j in range(4)]
    for j in range(3):
        if layer < cuts[j + 1]:
            return "L%d-%d" % (cuts[j], cuts[j + 1] - 1)
    return "L%d-%d" % (cuts[2], cuts[3] - 1)


def parse(key):
    l, k, i = key.split(".")
    return int(l), k, int(i)


def stratified_order(keys, n, seed=20261023, n_layer=12):
    """random_null.pick_targets with family-aware bands: equal numbers per (band, kind) cell, seeded, cells
    interleaved round-robin so any prefix covers every cell evenly."""
    cells = defaultdict(list)
    for k in keys:
        l, kind, _ = parse(k)
        cells[(band(l, n_layer), kind)].append(k)
    rng = random.Random(seed)
    per = max(1, n // max(1, len(cells)))
    picks = [rng.sample(sorted(cells[c]), min(per, len(cells[c]))) for c in sorted(cells)]
    out = [p[i] for i in range(max(map(len, picks), default=0)) for p in picks if i < len(p)]
    return out[:n]


def unique_in_order(seq):
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def _interleave(groups):
    return [g[i] for i in range(max(map(len, groups), default=0)) for g in groups if i < len(g)]


def panel(keys, per_cell, sub_per_cell, prior=(), seed=PANEL_SEED, n_layer=12):
    """The DAN-181 target panel (decided 2026-10-24): per (depth band, site kind) cell, `per_cell` targets. The first
    `sub_per_cell` of each cell are the extra-draw subsample: taken from `prior` (in its order; on TuringLLM the 63
    targets of the DAN-17 sweep run, 062/out_random, which were themselves a seeded 7-per-cell draw from the same
    list) where the cell has them, topped up by a seeded draw; the rest of the cell is a seeded draw from the cell's
    remaining keys. A union of two uniform draws, so every cell is still a uniform sample of its keys.
    Returns (order, sub_n): the subsample first, then the rest, each interleaved round-robin over the sorted cells, so
    `position < sub_n` marks the subsample (n_draws_for). A cell short of keys takes all it has."""
    keyset = set(keys)
    cells = defaultdict(list)
    for k in sorted(keyset):
        l, kind, _ = parse(k)
        cells[(band(l, n_layer), kind)].append(k)
    prior = [k for k in unique_in_order(prior) if k in keyset]
    rng = random.Random(seed)
    sub, rest = [], []
    for c in sorted(cells):
        pri = [k for k in prior if (band(parse(k)[0], n_layer), parse(k)[1]) == c][:sub_per_cell]
        left = [k for k in cells[c] if k not in set(pri)]
        top = rng.sample(left, min(max(0, sub_per_cell - len(pri)), len(left)))
        s = pri + top
        left = [k for k in left if k not in set(top)]
        sub.append(s)
        rest.append(rng.sample(left, min(max(0, per_cell - len(s)), len(left))))
    a = _interleave(sub)
    return a + _interleave(rest), len(a)


def n_draws_for(position, n_draws, sub_n, sub_draws):
    """Draws for the target at `position` of the (stratified, interleaved) order: the first sub_n get
    max(n_draws, sub_draws), the rest n_draws (the >= 2-draw subsample of the quantile test)."""
    return max(int(n_draws), int(sub_draws)) if position < int(sub_n) else int(n_draws)


def plan_jobs(refs, has_dan17, nulls, n_draws, splits, a1):
    """Every (arm, null, draw, split, alpha mode) of one target: fitted rows on each split, alpha = 1 rows on
    held-out strongest. wcm / dan17 are the references (one 'draw'); each null has n_draws draws."""
    jobs = []
    groups = ([("wcm", 1)] if "wcm" in refs else []) + ([("dan17", 1)] if "dan17" in refs and has_dan17 else [])
    groups += [(nl, int(n_draws)) for nl in nulls]
    for nl, nd in groups:
        for d in range(nd):
            jobs += [(nl, nl, d, s, "fit") for s in splits]
            if a1:
                jobs.append((nl + "_a1", nl, d, "held", "one"))
    return jobs


def rng_for(*parts):
    """A per-(target, null, draw) RNG, stable across processes, shards and resumes."""
    return random.Random(zlib.crc32("|".join(str(p) for p in parts).encode()))


# ----------------------------------------------------------------------------------------------- lags / deciles
def site_min_lag(top_acts, top_idx, anchors, d_sae):
    """[d_sae] int64: each latent's minimum anchor lag (anchor - position, positions <= anchor, activation > 0) over the
    batch; LAG_NONE where it never fires in a causal prefix. top_acts / top_idx [B, T, K] (post-Top-K codes)."""
    import torch
    B, T, K = top_acts.shape
    pos = torch.arange(T, device=top_acts.device)[None, :]
    anc = anchors.to(top_acts.device).long().view(-1, 1)[:B]
    lag = anc - pos                                                        # [B, T]
    valid = (lag >= 0)[..., None] & (top_acts > 0)                         # [B, T, K]
    lagk = lag[..., None].expand(B, T, K)
    big = torch.full_like(lagk, LAG_NONE)
    vals = torch.where(valid, lagk, big).reshape(-1)
    out = torch.full((int(d_sae),), LAG_NONE, dtype=torch.long, device=top_acts.device)
    out.scatter_reduce_(0, top_idx.reshape(-1).long(), vals.long(), reduce="amin", include_self=True)
    return out.cpu()


def freq_deciles(counts):
    """Per-site frequency deciles of a [d_sae] count vector: 0..9 over the latents with count > 0 (ascending rank,
    ties by index), -1 for never-active latents."""
    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    out = np.full(c.shape[0], -1, dtype=np.int64)
    alive = np.nonzero(c > 0)[0]
    if alive.size:
        order = alive[np.lexsort((alive, c[alive]))]                      # ascending count, ties by index
        out[order] = np.minimum(9, (10 * np.arange(order.size)) // order.size)
    return out


def rank_deciles(values):
    """{item: value} -> {item: decile 0..9}, ascending rank over all items (ties by item order)."""
    items = sorted(values, key=lambda x: (values[x], x))
    n = len(items)
    return {it: min(9, (10 * j) // n) for j, it in enumerate(items)}


def place_deciles(values, reference):
    """{item: value} -> {item: decile its value takes in `reference` (a list of values)}: min(9, 10 * r // n), r = the
    number of reference values below it."""
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    n = ref.size
    if not n or not values:
        return {}
    items = list(values)
    r = np.searchsorted(ref, np.asarray([values[x] for x in items], dtype=np.float64), side="left")
    return dict(zip(items, np.minimum(9, (10 * r) // n).astype(int).tolist()))


def member_bins(member_values, values):
    """Contribution bins defined by the production members' own distribution: edges at the members' 10th, 20th, ...
    90th percentiles give bins 0..9 (10% of members each); values below the members' minimum are bin -1 and above
    their maximum bin 10. {item: value} -> {item: bin}, for every item of `values` (members included).
    Matching on these bins reproduces the members' contribution distribution at decile resolution; deciles over the
    pool would not, because the production members sit almost entirely in the pool's top decile."""
    mv = np.sort(np.asarray(list(member_values), dtype=np.float64))
    if not mv.size or not values:
        return {}
    edges = np.percentile(mv, np.arange(10, 100, 10))
    items = list(values)
    v = np.asarray([values[x] for x in items], dtype=np.float64)
    b = np.searchsorted(edges, v, side="right")
    b = np.where(v < mv[0], -1, np.where(v > mv[-1], 10, b))
    return dict(zip(items, b.astype(int).tolist()))


def hist10(decs):
    """Counts per decile 0..9 (entries < 0, i.e. never-active, are counted separately by the caller)."""
    h = [0] * 10
    for d in decs:
        if 0 <= int(d) <= 9:
            h[int(d)] += 1
    return h


# ----------------------------------------------------------------------------------------------- samplers
def match_draw(ref_sites, cands, feat, rng, exclude=(), dist=None):
    """Draw, for every production member, one candidate at the same site, without replacement, choosing the
    candidate whose feature bin is nearest the member's (exact bin first; ties between bins and within a bin are
    uniform at random). ref_sites = {site: [members]}; cands = {site: [latents]} (the pool); feat(site, latent) ->
    tuple of ints (the bins; () = no matching, a uniform draw); exclude = {(site, latent)} never drawn (the
    production members). dist(a, b) -> sortable distance, default the tuple of absolute bin differences compared
    lexicographically (the first bin is matched first).

    Returns (support {site: [drawn]}, diag): n_ref, n_drawn, n_exact (same bin), short_sites (sites whose pool ran
    out), dist0 (sum of |first-bin difference| over drawn members), per-site pool sizes after exclusion, and by_bin
    {member first bin: [members, drawn, exact, summed |first-bin difference|]} (the match shortfall table)."""
    if dist is None:
        dist = lambda a, b: tuple(abs(x - y) for x, y in zip(a, b))
    excl = set(exclude)
    support, n_ref, n_drawn, n_exact, d0, short, pool_n = {}, 0, 0, 0, 0, [], {}
    by_bin = defaultdict(lambda: [0, 0, 0, 0])
    for site in sorted(ref_sites):
        members = sorted(int(m) for m in ref_sites[site])
        n_ref += len(members)
        rng.shuffle(members)                                               # greedy order is random, not by index
        buckets = defaultdict(list)
        for c in sorted(set(int(x) for x in cands.get(site, ()))):
            if (site, c) not in excl:
                buckets[tuple(feat(site, c))].append(c)
        pool_n[site] = sum(len(v) for v in buckets.values())
        drawn = []
        for m in members:
            fm = tuple(feat(site, m))
            bb = by_bin[str(fm[0]) if fm else "all"]
            bb[0] += 1
            if not buckets:
                short.append(site)
                continue                                                   # counted as members without a draw
            ds = {b: dist(fm, b) for b in buckets}
            best = min(ds.values())
            near = sorted(b for b in buckets if ds[b] == best)
            flat = [(b, j) for b in near for j in range(len(buckets[b]))]
            b, j = flat[rng.randrange(len(flat))]
            drawn.append(buckets[b].pop(j))
            if not buckets[b]:
                del buckets[b]
            n_exact += int(b == fm)
            d0 += abs(b[0] - fm[0]) if fm else 0
            bb[1] += 1
            bb[2] += int(b == fm)
            bb[3] += abs(b[0] - fm[0]) if fm else 0
        support[site] = drawn
        n_drawn += len(drawn)
    diag = dict(n_ref=n_ref, n_drawn=n_drawn, n_exact=n_exact, dist0=d0, short_sites=len(set(short)),
                pool_min=min(pool_n.values()) if pool_n else 0, pool_total=sum(pool_n.values()),
                by_bin={k: v for k, v in sorted(by_bin.items(), key=lambda kv: (kv[0] == "all", int(kv[0])
                                                                                 if kv[0] != "all" else 0))})
    return support, diag


def null_features(null, profile):
    """The feature function of a null: anchor = (frequency decile,); contrib = (contribution bin of the members'
    deciles, frequency decile); uniform = () (no matching; the anchor-active pool drawn uniformly, a diagnostic arm)."""
    fd = profile["fdec"]
    if null == "anchor":
        return lambda s, l: (int(fd[s].get(l, -1)),)
    if null == "contrib":
        cb = profile["cbin"]
        return lambda s, l: (int(cb.get((s, l), -1)), int(fd[s].get(l, -1)))
    if null == "uniform":
        return lambda s, l: ()
    raise ValueError("unknown null %r (anchor | contrib | uniform)" % null)


def anchor_pool(profile, win):
    """{site: [latents with anchor lag <= win]} from a profile."""
    return {s: sorted(l for l, g in lags.items() if g <= win) for s, lags in profile["lag"].items()}


EXCLUDE_DEFAULT = "contrib"     # Daniel, 2026-10-24 (final): members IN the anchor pool, OUT of the contrib pool


def parse_exclude(spec, nulls=("anchor", "contrib", "uniform")):
    """EXCLUDE_MEMBERS -> the set of nulls whose pool drops the production members. '0' / '' = none, '1' = every
    null, else a comma list of null names (default 'contrib': the contribution-matched null becomes a substitution
    test, since its top contribution bins hold almost only members; the anchor-active null keeps them)."""
    spec = (spec if spec is not None else EXCLUDE_DEFAULT).strip()
    if spec in ("", "0"):
        return set()
    if spec == "1":
        return set(nulls)
    out = {x.strip() for x in spec.split(",") if x.strip()}
    bad = out - set(nulls)
    if bad:
        raise ValueError("EXCLUDE_MEMBERS: unknown null(s) %s (known: %s)" % (sorted(bad), list(nulls)))
    return out


def draw_null(null, ref_sites, profile, win, rng, exclude_members=False):
    """One null draw. exclude_members=True drops the production members from the pool (2026-10-24 final: True for
    contrib, False for anchor; parse_exclude). diag gains n_overlap, the drawn production members."""
    pool = anchor_pool(profile, win)
    ref = {(s, int(m)) for s, v in ref_sites.items() for m in v}
    sup, diag = match_draw(ref_sites, pool, null_features(null, profile), rng, exclude=ref if exclude_members else set())
    diag["n_overlap"] = sum(1 for s, v in sup.items() for l in v if (s, int(l)) in ref)
    diag["exclude_members"] = bool(exclude_members)
    return sup, diag


def dan17_draw(key, ref_items, live):
    """DAN-17's random set exactly as 062/random_null.main draws it (draw 0): one RNG random.Random('<key>-null'); per
    production site in the circuit's member order (random_null.members_by_site), rng.sample(sorted(live[site]),
    min(count, |live[site]|)), live = latents active anywhere on the strongest-train contexts. ref_items = [(site,
    members)] in that order. Returns (support, diag) with random_null's n_ref / n_drawn / short_sites."""
    rng = random.Random("%s-null" % key)
    ref_items = list(ref_items)
    support = {s: rng.sample(sorted(live.get(s, ())), min(len(v), len(live.get(s, ())))) for s, v in ref_items}
    diag = dict(n_ref=sum(len(v) for _, v in ref_items), n_drawn=sum(len(v) for v in support.values()),
                short_sites=sum(1 for s, v in ref_items if len(live.get(s, ())) < len(v)))
    return support, diag


# ----------------------------------------------------------------------------------------------- coefficients
def softplus(x):
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def alpha_inert(steps, lr, weight_decay, schedule="constant"):
    """The coefficient of a latent that never gets a gradient: psi starts at softplus^-1(1) and AdamW's decoupled
    decay multiplies it by (1 - lr * wd) every step (Adam's own step is 0 for a zero gradient). Constant lr only (the
    protocol's); None otherwise."""
    if schedule != "constant":
        return None
    psi = math.log(math.expm1(1.0)) * (1.0 - float(lr) * float(weight_decay)) ** int(steps)
    return softplus(psi)


def alpha_stats(alphas, inert=None, tol=0.01):
    """Coefficient distribution of one circuit: quantiles, and the share that moved from the gradient-free
    trajectory (|alpha - inert| > tol; inert defaults to the circuit's modal value when it holds >= 5% of members)."""
    a = np.asarray([float(x) for x in alphas], dtype=np.float64)
    if a.size == 0:
        return dict(alpha_n=0)
    vals, cnt = np.unique(np.round(a, 6), return_counts=True)
    mode, mode_share = float(vals[cnt.argmax()]), float(cnt.max() / a.size)
    ref = inert if inert is not None else (mode if mode_share >= 0.05 else None)
    out = dict(alpha_n=int(a.size), alpha_median=float(np.median(a)), alpha_p10=float(np.percentile(a, 10)),
               alpha_p25=float(np.percentile(a, 25)), alpha_p75=float(np.percentile(a, 75)),
               alpha_p90=float(np.percentile(a, 90)), alpha_min=float(a.min()), alpha_max=float(a.max()),
               alpha_mode=mode, alpha_mode_share=mode_share, alpha_inert=ref)
    if ref is not None:
        out.update(alpha_moved=float(np.mean(np.abs(a - ref) > tol)), alpha_up=float(np.mean(a > ref + tol)),
                   alpha_down=float(np.mean(a < ref - tol)))
    return out


# ----------------------------------------------------------------------------------------------- set profiles
def set_profile(members, profile, win):
    """Summary of one member set ({site: [latents]}) against the target's profile: frequency- and contribution-decile
    histograms, median log10 corpus count, median contribution, the share that is anchor-active (lag <= win) and the
    share active anywhere in the causal prefix."""
    fd, cd, cnt, con, lag = profile["fdec"], profile["cdec"], profile["count"], profile["contrib"], profile["lag"]
    cb = profile.get("cbin", {})
    f, c, lc, cv, act, pre, dead, bins = [], [], [], [], 0, 0, 0, [0] * 12
    n = 0
    for s, v in members.items():
        for l in v:
            l = int(l); n += 1
            d = int(fd.get(s, {}).get(l, -1))
            dead += int(d < 0)
            f.append(d)
            c.append(int(cd.get((s, l), -1)))
            if (s, l) in cb:
                bins[int(cb[(s, l)]) + 1] += 1
            lc.append(math.log10(1 + float(cnt.get(s, {}).get(l, 0))))
            cv.append(float(con.get((s, l), 0.0)))
            g = lag.get(s, {}).get(l, LAG_NONE)
            act += int(g <= win); pre += int(g < LAG_NONE)
    if not n:
        return dict(n=0)
    return dict(n=n, fdec_hist=hist10(f), fdec_dead=dead, cdec_hist=hist10(c), log10_count_median=float(np.median(lc)),
                contrib_median=float(np.median(cv)), contrib_mean=float(np.mean(cv)),
                contrib_p25=float(np.percentile(cv, 25)), contrib_p75=float(np.percentile(cv, 75)), cbin_hist=bins,
                anchor_active=act / n, prefix_active=pre / n, fdec_mean=float(np.mean([x for x in f if x >= 0] or [np.nan])),
                cdec_mean=float(np.mean([x for x in c if x >= 0] or [np.nan])))


def coverage(members, profile):
    """Share of a member set active within each lag window (COVER_WINDOWS), plus 'prefix' (any p <= anchor)."""
    lag = profile["lag"]
    gs = [lag.get(s, {}).get(int(l), LAG_NONE) for s, v in members.items() for l in v]
    if not gs:
        return {}
    out = {"w%d" % w: float(np.mean([g <= w for g in gs])) for w in COVER_WINDOWS}
    out["prefix"] = float(np.mean([g < LAG_NONE for g in gs]))
    return out


# ----------------------------------------------------------------------------------------------- pass rule
def passes(r):
    """DAN-8 on the activation read: Z, A, C in [0.8, 1.5] and necessity >= 0.9; a missing score fails."""
    for c in HEAD:
        v = r.get(c)
        if v is None or (isinstance(v, float) and math.isnan(v)) or not (BAND[0] <= float(v) <= BAND[1]):
            return False
    v = r.get(NEC)
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and float(v) >= NEC_MIN


def train_subset(idx, n=TRAIN_EVAL_N):
    """n indices spread evenly over a training index list (stratified order kept): the training-split eval set."""
    idx = list(idx)
    if len(idx) <= n:
        return idx
    step = len(idx) / float(n)
    return [idx[int(j * step)] for j in range(n)]
