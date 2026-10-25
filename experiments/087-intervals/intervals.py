"""DAN-163: every headline number with its n and a 95% interval, traced to its data file (CPU only, no model).

Reads the row files the runs write and prints one table: point estimate, n, the 95% Wilson interval and the
target-level bootstrap interval (resample targets, B = 10,000, seeded per statistic), exact Clopper-Pearson bounds
and the rule of three for zero-pass nulls, and the source file(s) of every number. Writes

  OUTDIR/intervals.md    the table (every statistic, its macro name, both intervals, n, source)
  OUTDIR/intervals.tex   \\newcommand per statistic, for \\input in the paper: \\X = "46\\%\\ [45, 47]",
                         \\XN = "15{,}004", \\XFull = "46\\%\\ [45, 47], $n = 15{,}004$"
  OUTDIR/intervals.json  every statistic, machine-readable

Which interval the paper quotes (the macros; --quote): for a single pass rate, the WILSON interval. Targets are
independent draws and a pass is 0/1, so the target bootstrap of a proportion is the (resampled) Wald interval: it
agrees with Wilson to ~0.1 pp at n ~ 10^3-10^4 (both are printed), but it collapses to [0, 0] at k = 0 and
under-covers in small or extreme cells (attention at depth, alpha = 1 by band, public-model cells of ~60 targets).
Every statistic that is not a single proportion (median per-target size ratios, paired differences, ratios of pass
rates) quotes the target bootstrap (percentile, B = 10,000). Zero-pass nulls quote the exact one-sided 95%
Clopper-Pearson upper bound (1 - 0.05^(1/n) at k = 0), with the rule of three (3/n) beside it.

Sources (each optional; a missing one is skipped with a note):
  062 (today's paper):  --run062 (062/out_full/out: main/eval.*.jsonl, main/spec.*.jsonl, sweep_rows.csv)
                        --e062 (062/: out_alpha1/alpha1.jsonl, out_random_scale/random.jsonl,
                        out_{bands,depth,lowlam}/sweep/eval.shard*.jsonl, out_rescore_dedup/eval_dedup.jsonl)
  the 082 chain (2026-10-25 pod; WORK = its WORK dir, e.g. 082/smoke_final2 for the smoke):
    WORK/out_turingllm/rescore_ctx_v2/{main,spec,sweep,alpha1}   086 re-score on the rebuilt held-out sets
    WORK/out_turingllm/rescore_ctx_asrun1000/main                086 as-run re-score of the frozen 1,000 subset
    WORK/out_turingllm/harder_nulls/nulls.jsonl                  084 harder nulls
    WORK/out_turingllm/dan159/{table2_rows.csv, ref/rows*.jsonl} 083 method comparison (+ its 13-arm WCM/U re-score)
    WORK/out_<family>/{main,spec,sweep,alpha1,harder_nulls}      082 public runs (pythia-70m, gpt2-small)

  python experiments/087-intervals/intervals.py                     # today: the 062 outputs -> out/today
  python experiments/087-intervals/intervals.py --no-062 --work <WORK> --name pod   # the chain's outputs
  options: --B 10000  --seed 20261025  --quote wilson|bootstrap  --out DIR  --rescore-v2 DIR  --rescore-asrun DIR
           --asrun-targets FILE  --nulls084 LABEL=DIR  --dan159 DIR  --public FAMILY=DIR  --lowlam-shards paper|all
"""
import argparse
import glob
import json
import math
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E062 = ROOT / "experiments" / "062-h100-protocol-v1"
E086 = ROOT / "experiments" / "086-heldout-rebuild"
sys.path.insert(0, str(E062))
from family_table import band_label, family  # noqa: E402  (062's per-family bands; imports only src/model/families)

B_DEFAULT = 10_000
SEED = 20261025
Z95 = 1.959963984540054
CHUNK = 256                                   # bootstrap replicates per vectorised block (fixed: part of the seed)
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]
NEC = "phi_sup_blind_tk"
BAND = (0.8, 1.5)                             # DAN-8 (merge.py BAND / NEC_MIN)
NEC_MIN = 0.9
FLOOR = 8                                     # 086 heldout_rule.FLOOR: clean held-out strongest rows for the headline
KEEP = set(HEAD + [NEC, "seed", "held", "arm", "n", "vacuous_tk", "error", "skip", "n_held", "below_floor", "draw",
                   "split", "alpha", "pi", "target_faith_pre", "sibling_faith_median", "rank_clean", "weighted", "lam",
                   "phi_cf_alpha_blind_tk", "null"])
BAND_WORDS = ("Shallow", "Middle", "Deep")    # macro words for a family's three depth bands, in order


# ---------------------------------------------------------------------------------------------------- statistics
def wilson(k, n, z=Z95):
    """Two-sided 95% Wilson score interval for k successes of n (proportions)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def _beta_cdf_int(x, a, b):
    """Regularised incomplete beta I_x(a, b) for integer a, b >= 1: P(Binomial(a + b - 1, x) >= a)."""
    m = a + b - 1
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lx, l1 = math.log(x), math.log1p(-x)
    terms = [math.lgamma(m + 1) - math.lgamma(j + 1) - math.lgamma(m - j + 1) + j * lx + (m - j) * l1
             for j in range(a, m + 1)]
    mx = max(terms)
    return min(1.0, math.exp(mx) * sum(math.exp(t - mx) for t in terms))


def _beta_ppf(q, a, b):
    """Quantile of Beta(a, b) (integer a, b here). scipy when present, else bisection on the binomial identity."""
    try:
        from scipy.stats import beta
        return float(beta.ppf(q, a, b))
    except ImportError:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _beta_cdf_int(mid, a, b) < q else (lo, mid)
        return (lo + hi) / 2


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided (1 - alpha) Clopper-Pearson interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else _beta_ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


def cp_upper(k, n, alpha=0.05):
    """Exact ONE-sided (1 - alpha) Clopper-Pearson upper bound; at k = 0 it is 1 - alpha^(1/n)."""
    if n == 0:
        return float("nan")
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    return _beta_ppf(1 - alpha, k + 1, n - k)


def rule_of_three(n):
    """The one-sided 95% upper bound 3/n for 0 events in n (Hanley & Lippman-Hand 1983)."""
    return 3.0 / n if n else float("nan")


def stat_seed(key, seed=SEED):
    """Each statistic's own RNG seed: adding or removing a statistic never changes another's interval."""
    return (seed ^ zlib.crc32(key.encode())) & 0xFFFFFFFF


def bootstrap(fn, n, B=B_DEFAULT, seed=SEED, chunk=CHUNK):
    """Target-level bootstrap: B resamples of n targets with replacement; fn(idx [c, n] int array) -> c statistics.
    Deterministic for (seed, B, chunk). Returns the B replicate statistics."""
    rng = np.random.default_rng(seed)
    out = []
    done = 0
    while done < B:
        c = min(chunk, B - done)
        out.append(np.asarray(fn(rng.integers(0, n, size=(c, n))), float))
        done += c
    return np.concatenate(out)


def pct_ci(reps):
    reps = np.asarray(reps, float)
    reps = reps[np.isfinite(reps)]
    if not len(reps):
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return (float(lo), float(hi))


def boot_mean(x, B=B_DEFAULT, seed=SEED):
    x = np.asarray(x, float)
    return pct_ci(bootstrap(lambda idx: x[idx].mean(axis=1), len(x), B, seed)) if len(x) else (np.nan, np.nan)


def boot_median(x, B=B_DEFAULT, seed=SEED):
    x = np.asarray(x, float)
    return pct_ci(bootstrap(lambda idx: np.median(x[idx], axis=1), len(x), B, seed)) if len(x) else (np.nan, np.nan)


def boot_paired_diff(a, b, B=B_DEFAULT, seed=SEED):
    """Paired bootstrap of mean(b - a): the same targets resampled for both arms."""
    d = np.asarray(b, float) - np.asarray(a, float)
    return boot_mean(d, B, seed)


def boot_ratio_of_means(num, den, B=B_DEFAULT, seed=SEED):
    """Paired bootstrap of mean(num) / mean(den) (e.g. the share of targets passing under one method over another)."""
    a, b = np.asarray(num, float), np.asarray(den, float)
    if not len(a):
        return (np.nan, np.nan)

    def fn(idx):
        d = b[idx].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(d > 0, a[idx].mean(axis=1) / d, np.nan)
    return pct_ci(bootstrap(fn, len(a), B, seed))


# ------------------------------------------------------------------------------------------------- formatting
NUM_WORDS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def macro_name(*parts):
    """A LaTeX-safe camelCase macro name (letters only) from key parts; digits become words."""
    words = []
    for p in parts:
        tok, cur = [], ""
        for ch in str(p):
            if ch.isalpha():
                cur += ch
            else:
                if cur:
                    tok.append(cur); cur = ""
                if ch.isdigit():
                    tok.append(NUM_WORDS[int(ch)])
                elif ch == ".":
                    tok.append("Pt")
        if cur:
            tok.append(cur)
        words += [t[0].upper() + t[1:] for t in tok if t]
    s = "".join(words)
    return s[0].lower() + s[1:] if s else s


def thousands(n):
    return "{:,}".format(int(n))


def _dec(p, dec, lo=None, hi=None):
    """Decimals of a percentage: one below 10% (1.5%, 2.1%) or when the interval is narrower than 2 pp (so the
    headline reads 46.4% [45.6, 47.2], not 46% [46, 47]); else none."""
    if dec is not None:
        return dec
    narrow = lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) and (hi - lo) < 0.02
    return 1 if (p < 0.10 or p > 0.995 or narrow) else 0


def fmt_pct(x, dec):
    return ("%." + str(dec) + "f") % (100 * x)


class Stats:
    """The collected statistics, in order; each a dict (see prop / ratio / diff / median)."""

    def __init__(self, B=B_DEFAULT, seed=SEED, quote="wilson"):
        self.rows, self.notes, self.B, self.seed, self.quote = [], [], B, seed, quote
        self.names, self.skipped = set(), []

    def _add(self, d, macro=True):
        if not d["n"] or not np.isfinite(d["point"]):        # an empty cell (or 0 / 0): nothing to report
            self.skipped.append("%s %s%s" % (d["section"], d["prefix"] + "." if d["prefix"] else "", d["key"]))
            return None
        d["macro"] =macro_name(d["prefix"], *d["key"].split(".")) if macro else ""
        if d["macro"]:
            if d["macro"] in self.names:
                raise ValueError("duplicate macro %s (key %s)" % (d["macro"], d["key"]))
            self.names.add(d["macro"])
        self.rows.append(d)
        return d

    def prop(self, section, prefix, key, label, x, source, dec=None, macro=True, null=False):
        """A pass rate (or any share of targets): x = one 0/1 value per target."""
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        n, k = len(x), int(x.sum())
        p = k / n if n else float("nan")
        boot = boot_mean(x, self.B, stat_seed(prefix + key, self.seed)) if n else (np.nan, np.nan)
        d = dict(section=section, prefix=prefix, key=key, label=label, type="prop", k=k, n=n, point=p,
                 wilson=wilson(k, n), boot=boot, cp=clopper_pearson(k, n) if n else (np.nan, np.nan),
                 cp_upper=cp_upper(k, n), rule3=rule_of_three(n) if k == 0 else float("nan"), source=source,
                 dec=dec, null=null)
        return self._add(d, macro)

    def median(self, section, prefix, key, label, x, source, unit="", dec=0, macro=True):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        boot = boot_median(x, self.B, stat_seed(prefix + key, self.seed)) if len(x) else (np.nan, np.nan)
        return self._add(dict(section=section, prefix=prefix, key=key, label=label, type="median", n=len(x),
                              point=float(np.median(x)) if len(x) else np.nan, boot=boot, source=source, unit=unit,
                              dec=dec), macro)

    def diff(self, section, prefix, key, label, a, b, source, macro=True, dec=1):
        """Paired difference of shares, mean(b - a) in percentage points, over the same targets."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = len(a)
        boot = boot_paired_diff(a, b, self.B, stat_seed(prefix + key, self.seed)) if n else (np.nan, np.nan)
        return self._add(dict(section=section, prefix=prefix, key=key, label=label, type="diff", n=n,
                              point=float((b - a).mean()) if n else np.nan, a=float(a.mean()) if n else np.nan,
                              b=float(b.mean()) if n else np.nan, boot=boot, source=source, dec=dec,
                              up=int(((a == 0) & (b == 1)).sum()), down=int(((a == 1) & (b == 0)).sum())), macro)

    def ratio(self, section, prefix, key, label, num, den, source, macro=True, dec=2):
        """Ratio of two shares over the same targets, mean(num) / mean(den), paired bootstrap."""
        num, den = np.asarray(num, float), np.asarray(den, float)
        boot = boot_ratio_of_means(num, den, self.B, stat_seed(prefix + key, self.seed))
        pt = float(num.mean() / den.mean()) if len(den) and den.mean() > 0 else np.nan
        return self._add(dict(section=section, prefix=prefix, key=key, label=label, type="ratio", n=len(num),
                              point=pt, boot=boot, source=source, dec=dec, unit="x"), macro)

    def note(self, text):
        self.notes.append(text)

    # ---- text
    def quoted(self, d):
        """(lo, hi, method) that the paper quotes for this statistic."""
        if d["type"] == "prop":
            if d["k"] == 0 and d["n"]:
                return (0.0, d["cp_upper"], "one-sided CP")
            if self.quote == "bootstrap":
                return (*d["boot"], "bootstrap")
            return (*d["wilson"], "Wilson")
        return (*d["boot"], "bootstrap")

    def text(self, d, tex=False):
        lo, hi, _ = self.quoted(d)
        pc = r"\%" if tex else "%"
        sp = r"\ " if tex else " "
        if d["type"] == "prop":
            if not d["n"]:
                return "-"
            if d["k"] == 0:
                ub = d["cp_upper"]
                dec = 2 if ub < 0.01 else 1
                return "0/%s%s(%s %s%s)" % (thousands(d["n"]) if not tex else thousands(d["n"]).replace(",", "{,}"),
                                            sp, r"$\leq$" if tex else "<=", fmt_pct(ub, dec), pc)
            dec = _dec(d["point"], d["dec"], lo, hi)
            return "%s%s%s[%s, %s]" % (fmt_pct(d["point"], dec), pc, sp, fmt_pct(lo, dec), fmt_pct(hi, dec))
        if d["type"] == "diff":
            dec = d["dec"]
            f = lambda v: ("%+." + str(dec) + "f") % (100 * v)
            return "%s pp%s[%s, %s]" % (f(d["point"]), sp, f(lo), f(hi))
        if d["type"] in ("ratio", "median"):
            dec = d["dec"]
            f = lambda v: ("{:,.%df}" % dec).format(v) if np.isfinite(v) else "-"
            u = d.get("unit", "")
            ux = (r"$\times$" if tex else "x") if u == "x" else ""
            s = "%s%s%s[%s, %s]" % (f(d["point"]), ux, sp, f(lo), f(hi))
            return s.replace(",", "{,}").replace("{,} ", ", ") if tex else s
        return "-"

    def full(self, d, tex=False):
        n = d["n"]
        if tex:
            return "%s, $n = %s$" % (self.text(d, True), thousands(n).replace(",", "{,}"))
        return "%s, n = %s" % (self.text(d), thousands(n))


# --------------------------------------------------------------------------------------------------- readers
def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def row_files(d, stem):
    """stem.jsonl plus stem.<anything>.jsonl (the driver's shard files, eval.shard0.jsonl ...)."""
    d = Path(d)
    fs = sorted(set(glob.glob(str(d / ("%s.jsonl" % stem))) + glob.glob(str(d / ("%s.*.jsonl" % stem)))))
    return [Path(f) for f in fs]


def read_jsonl(files, keep=KEEP):
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001 (a torn last line of a live shard)
                    continue
                rows.append({k: v for k, v in r.items() if k in keep} if keep else r)
    return pd.DataFrame(rows)


def src_of(files, extra=""):
    files = list(files)
    if not files:
        return "(none)"
    if len(files) == 1:
        s = rel(files[0])
    else:
        d = rel(files[0].parent)
        stems = sorted({f.name.split(".")[0] for f in files})
        s = "%s/%s.*.jsonl (%d files)" % (d, "|".join(stems), len(files))
    return s + ("; " + extra if extra else "")


def clean(df, skip=True):
    if df.empty:
        return df
    if "error" in df:
        df = df[df["error"].isna()]
    if skip and "skip" in df:
        df = df[df["skip"].isna()]
    return df


def passes(df):
    """The DAN-8 rule on the activation read (= 062 merge.passes): Z, A, C in [0.8, 1.5] and necessity >= 0.9; a
    missing score fails. Apply after removing vacuous rows."""
    for c in HEAD + [NEC]:
        if c not in df:
            df[c] = np.nan
    x = df[HEAD].apply(pd.to_numeric, errors="coerce")
    inb = ((x >= BAND[0]) & (x <= BAND[1])).all(axis=1)
    return (inb & (pd.to_numeric(df[NEC], errors="coerce") >= NEC_MIN)).astype(bool)


def vacuous(df):
    if "vacuous_tk" not in df:
        return pd.Series(False, index=df.index)
    v = df["vacuous_tk"]
    return v.astype(object).where(v.notna(), False).astype(bool)


def bands_of(fam):
    return [(band_label(lo, hi), lo, hi) for lo, hi in fam["bands"]]


def tag(df, fam):
    df = df.copy()
    df["layer"] = [int(s.split(".")[0]) for s in df.seed]
    df["kind"] = [s.split(".", 1)[1].rsplit(".", 1)[0] for s in df.seed]
    bl = bands_of(fam)
    df["band"] = [next((b for b, _, hi in bl if l <= hi), bl[-1][0]) for l in df.layer]
    return df


def driver_main(out, fam):
    """062 driver main rows (OUT/main/eval.*.jsonl): held-out strongest and mid-band frames, one row per target, vacuous
    removed, with pass, layer, kind, band. Returns (strong, mid, n_vacuous_strong, files)."""
    files = row_files(Path(out) / "main", "eval")
    ev = clean(read_jsonl(files))
    if ev.empty:
        return None, None, 0, files
    ev = ev.drop_duplicates(["seed", "held"], keep="last")
    res = []
    for h in ("strong", "mid"):
        d = ev[ev.held == h].copy()
        vac = vacuous(d)
        d = d[~vac].copy()
        d["pass"] = passes(d)
        res.append((tag(d, fam), int(vac.sum())))
    return res[0][0], res[1][0], res[0][1], files


def amp_flags(spec):
    """Per target: flagged as an amplifier under any ablation method (056 / merge.py: target_faith_pre >= 0.5 and
    sibling_faith_median >= 0.8 x target_faith_pre)."""
    sp = clean(spec, skip=True)
    if sp.empty:
        return pd.Series(dtype=bool)
    sp = sp.drop_duplicates(["seed", "pi"], keep="last")
    t, s = pd.to_numeric(sp.target_faith_pre, errors="coerce"), pd.to_numeric(sp.sibling_faith_median, errors="coerce")
    sp = sp.assign(flag=(t >= 0.5) & (s >= 0.8 * t))
    return sp.groupby("seed").flag.any()


def rescore_classify(files, keys=("seed",)):
    """086 rescore rows through 086 passrates.classify (the owner's below-floor / rejected / vacuous bookkeeping)."""
    sys.path.insert(0, str(E086))
    import passrates  # noqa: E402
    rows = read_jsonl(files)
    return passrates.classify(rows.to_dict("records"), keys) if len(rows) else pd.DataFrame()


# ------------------------------------------------------------------------------------------------- sections
def band_word(fam, b):
    return BAND_WORDS[[x for x, _, _ in bands_of(fam)].index(b)]


def pass_breakdown(S, section, prefix, d, fam, source, what="pass", worst=None, layers=True):
    """Pass rates of one frame (one row per target, 'pass' bool, band / kind / layer): overall, by depth band, by
    site kind, by band x kind, by layer (md only). worst: a second frame for the worst case (below floor = fail)."""
    S.prop(section, prefix, what + ".all", "all targets", d["pass"], source)
    bl = [b for b, _, _ in bands_of(fam)]
    kinds = [k for k in fam["kinds"] if k in set(d.kind)] + sorted(set(d.kind) - set(fam["kinds"]))
    for b in bl:
        S.prop(section, prefix, "%s.%s" % (what, band_word(fam, b)), "layers %s" % b[1:], d[d.band == b]["pass"], source)
    for k in kinds:
        S.prop(section, prefix, "%s.%s" % (what, k), "site kind %s" % k, d[d.kind == k]["pass"], source)
    for b in bl:
        for k in kinds:
            g = d[(d.band == b) & (d.kind == k)]
            if len(g):
                S.prop(section, prefix, "%s.%s.%s" % (what, band_word(fam, b), k), "layers %s, %s" % (b[1:], k),
                       g["pass"], source)
    if layers:
        for l, g in d.groupby("layer"):
            S.prop(section, prefix, "%s.layer%d" % (what, l), "layer %d" % l, g["pass"], source)
    if worst is not None:
        S.prop(section, prefix, what + ".worst.all", "worst case (below-floor targets counted as fails)",
               worst["pass"], source)
        for b in bl:
            S.prop(section, prefix, "%s.worst.%s" % (what, band_word(fam, b)), "worst case, layers %s" % b[1:],
                   worst[worst.band == b]["pass"], source)


def amplifier_section(S, section, prefix, strong, amp, source):
    amp = amp.to_dict()
    a = np.array([bool(amp.get(s, False)) for s in strong.seed])
    p = strong["pass"].values
    S.prop(section, prefix, "amp.all", "flagged as amplifier, all circuits (no spec row = not flagged)", a, source)
    for k in sorted(set(strong.kind)):
        m = (strong.kind == k).values
        S.prop(section, prefix, "amp." + k, "flagged as amplifier, %s" % k, a[m], source)
    S.prop(section, prefix, "amp.passing", "flagged as amplifier, among passing circuits", a[p], source)
    S.prop(section, prefix, "amp.passNoFlag", "passes AND not flagged, all circuits", p & ~a, source)
    S.note("%s: %d of %d strongest targets have a specificity row (missing rows count as not flagged, as paper_main)."
           % (section, sum(s in amp for s in strong.seed), len(strong)))


def alpha1_section(S, section, prefix, a1, fitted, fam, source):
    """alpha = 1 rows (one per target, vacuous removed) vs the fitted circuits' pass on the same targets."""
    S.prop(section, prefix, "aone.all", "alpha = 1, all targets", a1["pass"], source)
    for b, _, _ in bands_of(fam):
        S.prop(section, prefix, "aone." + band_word(fam, b), "alpha = 1, layers %s" % b[1:], a1[a1.band == b]["pass"],
               source)
    for k in sorted(set(a1.kind)):
        S.prop(section, prefix, "aone." + k, "alpha = 1, %s" % k, a1[a1.kind == k]["pass"], source)
    j = a1.set_index("seed")[["pass", "band"]].join(fitted.rename("fit"), how="inner")
    j = j[j.fit.notna()]
    S.prop(section, prefix, "aone.fitted", "fitted circuits on the same targets", j.fit.astype(float), source)
    for b, _, _ in bands_of(fam):
        g = j[j.band == b]
        S.prop(section, prefix, "aone.fitted." + band_word(fam, b), "fitted, same targets, layers %s" % b[1:],
               g.fit.astype(float), source)
    S.diff(section, prefix, "aone.drop", "paired: alpha = 1 minus fitted (same targets)", j.fit.astype(float),
           j["pass"].astype(float), source)


def null_section(S, section, prefix, rn, fitted, fam, source, label="random circuits"):
    S.prop(section, prefix, "all", "%s, all targets" % label, rn["pass"], source, null=True)
    for b, _, _ in bands_of(fam):
        S.prop(section, prefix, band_word(fam, b), "%s, layers %s" % (label, b[1:]), rn[rn.band == b]["pass"], source,
               null=True)
    for k in sorted(set(rn.kind)):
        S.prop(section, prefix, k, "%s, %s" % (label, k), rn[rn.kind == k]["pass"], source, null=True)
    if fitted is not None:
        f = rn.seed.map(fitted)
        S.prop(section, prefix, "fitted", "fitted circuits on the same targets", f[f.notna()].astype(float), source)


def per_target_budget(rows):
    """Per target: smallest passing size (t) and whether any circuit passes (o), else the largest size fitted
    (062 paper_main.per_target)."""
    g = rows.groupby("seed")
    best = rows[rows["pass"]].groupby("seed").n.min()
    big = g.n.max()
    return pd.DataFrame({"t": best.reindex(big.index).fillna(big), "o": big.index.isin(best.index)})


def sweep_section(S, section, prefix, sw, fam, source, prod_lam=1e-3, low=None, low_source=""):
    """The coefficient claims on the sweep. sw: one row per (seed, arm) with weighted, lam, n, pass, band (vacuous
    rows kept with pass False, as merge.py's sweep_rows.csv). low: extra WCM rows at weaker penalties (the 8-penalty
    grid of the budget figure; vacuous rows removed, as paper_main.figure_budget)."""
    bl = [b for b, _, _ in bands_of(fam)]
    f = sw[np.isclose(sw.lam, prod_lam)]
    for b in bl:
        for w, nm in ((True, "wcm"), (False, "unw")):
            g = f[(f.band == b) & (f.weighted == w)]
            S.prop(section, prefix, "fixed.%s.%s" % (nm, band_word(fam, b)), "lambda %g, %s, layers %s" % (
                prod_lam, "WCM" if w else "unweighted", b[1:]), g["pass"], source)
        p = f[f.band == b].pivot_table(index="seed", columns="weighted", values="pass", aggfunc="last").dropna()
        if True in p and False in p:
            S.diff(section, prefix, "fixed.diff." + band_word(fam, b), "lambda %g: unweighted minus WCM, paired, layers %s"
                   % (prod_lam, b[1:]), p[True].astype(float), p[False].astype(float), source)
    for w, nm in ((True, "wcm"), (False, "unw")):
        for lam, g in sw[sw.weighted == w].groupby("lam"):
            S.prop(section, prefix, "lam.%s.%s" % (nm, ("%g" % lam).replace("-", "m")),
                   "%s, lambda %g, all bands (median %.0f latents)" % ("WCM" if w else "unweighted", lam, g.n.median()),
                   g["pass"], source)
    src = source + ((" + " + low_source) if low is not None else "")
    for b in bl:
        bw = band_word(fam, b)
        rows_u = sw[(~sw.weighted) & (sw.band == b)][["seed", "n", "pass"]]
        rows_w = sw[sw.weighted & (sw.band == b)][["seed", "n", "pass"]]
        if low is not None and len(low):
            rows_w = pd.concat([rows_w, low[(low.band == b) & low.seed.isin(set(rows_w.seed))][["seed", "n", "pass"]]])
        if not len(rows_u) or not len(rows_w):
            continue
        pw, pu = per_target_budget(rows_w), per_target_budget(rows_u)
        j = pw.join(pu, lsuffix="_w", rsuffix="_u", how="inner")
        S.prop(section, prefix, "ever.wcm." + bw, "passes at some penalty, WCM, layers %s" % b[1:], j.o_w, src)
        S.prop(section, prefix, "ever.unw." + bw, "passes at some penalty, unweighted, layers %s" % b[1:], j.o_u, src)
        S.ratio(section, prefix, "ever.ratio." + bw, "share passing at all, unweighted / WCM, layers %s" % b[1:],
                j.o_u.astype(float), j.o_w.astype(float), src)
        S.diff(section, prefix, "ever.diff." + bw, "share passing at all, unweighted minus WCM, layers %s" % b[1:],
               j.o_w.astype(float), j.o_u.astype(float), src)
        S.prop(section, prefix, "kilo.wcm." + bw, "passing circuit of <= 1,000 latents, WCM, layers %s" % b[1:],
               (j.o_w & (j.t_w <= 1000)), src)
        S.prop(section, prefix, "kilo.unw." + bw, "passing circuit of <= 1,000 latents, unweighted, layers %s" % b[1:],
               (j.o_u & (j.t_u <= 1000)), src)
        both = j[j.o_w & j.o_u]
        S.median(section, prefix, "ratio." + bw, "median per-target size ratio unweighted / WCM (smallest passing "
                 "circuits; targets passing under both), layers %s" % b[1:], both.t_u / both.t_w, src, unit="x", dec=1)
        S.median(section, prefix, "size.wcm." + bw, "median smallest passing WCM circuit (same targets), layers %s"
                 % b[1:], both.t_w, src)
        S.median(section, prefix, "size.unw." + bw, "median smallest passing unweighted circuit (same targets), "
                 "layers %s" % b[1:], both.t_u, src)


def arm_lam(arm):
    """'W_0.001' -> (True, 0.001); 'U_1e-05' -> (False, 1e-05); else (None, None)."""
    if not isinstance(arm, str) or arm[:2] not in ("W_", "U_") or "_s" in arm[1:]:
        return None, None
    try:
        return arm[0] == "W", float(arm[2:])
    except ValueError:
        return None, None


def sweep_from_rows(df, fam, drop_vacuous=False):
    """Sweep-style rows (W_/U_ arms) -> one row per (seed, arm) with weighted, lam, n, pass, band."""
    df = clean(df)
    if df.empty:
        return df
    df = df.drop_duplicates(["seed", "arm"], keep="last").copy()
    wl = [arm_lam(a) for a in df.arm]
    df["weighted"], df["lam"] = [w for w, _ in wl], [l for _, l in wl]
    df = df[df.lam.notna()].copy()
    df["weighted"] = df.weighted.astype(bool)
    vac = vacuous(df)
    df["pass"] = passes(df) & ~vac
    if drop_vacuous:
        df = df[~vac].copy()
    df["n"] = pd.to_numeric(df.n, errors="coerce")
    return tag(df, fam)


def leakage_section(S, section, prefix, asrun, v2, sub, fam, source):
    """The paired held-out leakage effect (rebuilt minus as-run) on the frozen subset; 086 passrates.paired's pairing
    (circuit in both, scored and non-vacuous as run, not vacuous rebuilt)."""
    for held in ("strong", "mid"):
        a, b = asrun[asrun.held == held], v2[v2.held == held]
        if sub is not None:
            a, b = a[a.seed.isin(sub)], b[b.seed.isin(sub)]
        if not len(a) or not len(b):
            continue
        j = a.set_index("seed")[["pass", "vac", "rejected", "scored", "band", "kind"]].join(
            b.set_index("seed")[["pass", "vac", "rejected", "scored", "below"]], lsuffix="_a", rsuffix="_b",
            how="inner")
        j = j[~j.rejected_a & ~j.rejected_b & j.scored_a & ~j.vac_a & ~(j.scored_b & j.vac_b)]
        h = j[~j.below]
        hw = "" if held == "strong" else "mid."
        lab = "held-out %s" % ("strongest" if held == "strong" else "mid-band")
        S.prop(section, prefix, hw + "asrun", "%s, as run (headline pairs)" % lab, h.pass_a.astype(float), source)
        S.prop(section, prefix, hw + "rebuilt", "%s, rebuilt (headline pairs)" % lab, h.pass_b.astype(float), source)
        S.diff(section, prefix, hw + "effect", "%s: rebuilt minus as run, paired (headline pairs)" % lab,
               h.pass_a.astype(float), h.pass_b.astype(float), source)
        worst_b = (j.pass_b & ~j.below).astype(float)
        S.diff(section, prefix, hw + "effect.worst", "%s: rebuilt (below floor = fail) minus as run, all pairs" % lab,
               j.pass_a.astype(float), worst_b, source)
        bel = j[j.below]
        S.prop(section, prefix, hw + "below", "%s: share of pairs below the floor (< %d clean rows)" % (lab, FLOOR),
               j.below.astype(float), source)
        if len(bel) and bel.scored_b.any():
            S.prop(section, prefix, hw + "below.rebuilt", "%s: below-floor targets' own rebuilt pass rate (scored)"
                   % lab, bel[bel.scored_b].pass_b.astype(float), source)
            S.prop(section, prefix, hw + "below.asrun", "%s: below-floor targets' as-run pass rate" % lab,
                   bel.pass_a.astype(float), source)
        if held == "strong":
            for bnd, g in h.groupby("band"):
                S.diff(section, prefix, "effect." + band_word(fam, bnd), "%s: rebuilt minus as run, layers %s"
                       % (lab, bnd[1:]), g.pass_a.astype(float), g.pass_b.astype(float), source)
            for (bnd, k), g in h.groupby(["band", "kind"]):
                S.diff(section, prefix, "effect.%s.%s" % (band_word(fam, bnd), k), "%s: rebuilt minus as run, "
                       "layers %s, %s" % (lab, bnd[1:], k), g.pass_a.astype(float), g.pass_b.astype(float), source,
                       macro=False)


def rebuilt_frame(cls, held, fam):
    """086 classify frame -> (headline frame: scored, not below floor, not vacuous; worst frame: + below-floor as
    fails), tagged with the family's bands."""
    d = cls[(cls.held == held) & ~cls.rejected] if "held" in cls else cls[~cls.rejected]
    head = d[d.scored & ~d.below & ~d.vac].copy()
    worst = pd.concat([head, d[d.below].assign(**{"pass": False})])
    return tag(head[["seed", "pass"]], fam), tag(worst[["seed", "pass"]], fam), int(d.below.sum())


def nulls084_section(S, section, prefix, d84, fam, source, fitted=None):
    """084 harder nulls: per arm, draw 0, held-out strongest, vacuous removed (report.py table 1)."""
    df = clean(read_jsonl(row_files(d84, "nulls")))
    if df.empty:
        S.note("%s: no rows in %s" % (section, rel(d84)))
        return
    for c in ("draw", "split"):
        if c not in df:
            df[c] = 0 if c == "draw" else "held"
    df = df.drop_duplicates(["seed", "arm", "draw", "split"], keep="last")
    df = df[(df.split == "held") & (pd.to_numeric(df.draw, errors="coerce") == 0)].copy()
    df = df[~vacuous(df)].copy()
    df["pass"] = passes(df)
    df = tag(df, fam)
    order = ["wcm", "wcm_a1", "dan17", "dan17_a1", "anchor", "anchor_a1", "contrib", "contrib_a1", "uniform",
             "uniform_a1"]
    arms = [a for a in order if a in set(df.arm)] + sorted(set(df.arm) - set(order))
    for a in arms:
        g = df[df.arm == a]
        null = a not in ("wcm",)
        S.prop(section, prefix, a.replace("_", "") + ".all", "arm %s, all targets (draw 0)" % a, g["pass"], source,
               null=null)
        for b, _, _ in bands_of(fam):
            S.prop(section, prefix, "%s.%s" % (a.replace("_", ""), band_word(fam, b)), "arm %s, layers %s" % (
                a, b[1:]), g[g.band == b]["pass"], source, null=null)


def dan159_section(S, section, prefix, csv, fam, source):
    """083 table2_rows.csv: every per-target row of Table 2 with its method row label; pass rates per row and band,
    and the paired difference from the WCM row on shared targets."""
    t = pd.read_csv(csv, low_memory=False)
    if t.empty:
        return
    vac = vacuous(t)
    t = t[~vac].copy()
    p = passes(t)
    if "passes" in t:
        mism = int((p != t["passes"].astype(bool)).sum())
        if mism:
            S.note("%s: %d rows where table2's own 'passes' differs from the DAN-8 rule recomputed here" % (section,
                                                                                                         mism))
    t["pass"] = p
    t = tag(t, fam)
    wcm = None
    for i, ((sec, row), g) in enumerate(t.groupby(["section", "row"], sort=False)):
        g = g.drop_duplicates("seed", keep="last")
        key = "m%d" % i
        S.prop(section, prefix, key + ".all", "%s | %s" % (sec, row), g["pass"], source)
        for b, _, _ in bands_of(fam):
            S.prop(section, prefix, "%s.%s" % (key, band_word(fam, b)), "%s | layers %s" % (row, b[1:]),
                   g[g.band == b]["pass"], source)
        if wcm is None and str(row).startswith("WCM"):
            wcm = g.set_index("seed")["pass"]
        elif wcm is not None:
            j = g.set_index("seed")["pass"].rename("m").to_frame().join(wcm.rename("w"), how="inner")
            if len(j):
                S.diff(section, prefix, key + ".vsWcm", "%s minus WCM, paired" % row, j.w.astype(float),
                       j.m.astype(float), source)


# ---------------------------------------------------------------------------------------------------- runners
def run_062(S, run062, e062, lowlam_shards="paper"):
    fam = family("turingllm")
    strong, mid, nvac, files = driver_main(run062, fam)
    if strong is None:
        S.note("062: no main eval rows under %s" % rel(run062))
        return None
    src = src_of(files, "held = strong, vacuous removed (%d)" % nvac)
    sec = "062 production run (TuringLLM, as-run held-out sets)"
    pass_breakdown(S, sec, "", strong, fam, src)
    S.prop(sec, "", "pass.midHeld", "held-out mid-band contexts, all targets", mid["pass"],
           src_of(files, "held = mid, vacuous removed"))
    spf = row_files(Path(run062) / "main", "spec")
    if spf:
        amplifier_section(S, "062 specificity (056, held-out strongest)", "", strong, amp_flags(read_jsonl(spf)),
                          src_of(spf))
    fitted = strong.set_index("seed")["pass"].astype(float)
    a1f = Path(e062) / "out_alpha1" / "alpha1.jsonl"
    if a1f.exists():
        a1 = clean(read_jsonl([a1f])).drop_duplicates("seed", keep="last")
        a1 = a1[~vacuous(a1)].copy()
        a1["pass"] = passes(a1)
        alpha1_section(S, "062 alpha = 1 (same nodes, every coefficient 1)", "", tag(a1, fam), fitted, fam,
                       src_of([a1f], "vacuous removed"))
    rnf = Path(e062) / "out_random_scale" / "random.jsonl"
    if rnf.exists():
        rn = clean(read_jsonl([rnf]))
        rn = rn[~vacuous(rn)].copy()
        rn["pass"] = passes(rn)
        null_section(S, "062 random null (DAN-17, size-matched, coefficients fitted)", "null", tag(rn, fam), fitted,
                     fam, src_of([rnf], "vacuous removed"))
    swf = Path(run062) / "sweep_rows.csv"
    if swf.exists():
        sw = pd.read_csv(swf, low_memory=False)
        sw = sw.rename(columns={"passes": "pass"})
        sw["pass"] = sw["pass"].astype(bool)
        sw = tag(sw, fam)
        lows = {}
        for mode in ("paper", "all"):
            fs = []
            for dname in ("out_bands", "out_depth", "out_lowlam"):
                dd = Path(e062) / dname / "sweep"
                if mode == "all":
                    fs += row_files(dd, "eval")
                elif (dd / "eval.shard0.jsonl").exists():
                    fs.append(dd / "eval.shard0.jsonl")
            low = sweep_from_rows(read_jsonl(fs), fam, drop_vacuous=True)
            lows[mode] = (low[low.weighted] if len(low) else low, fs)
        low, fs = lows[lowlam_shards]
        sweep_section(S, "062 sweep: WCM vs unweighted (190 targets; budget = 8 WCM / 5 unweighted penalties)", "sw",
                      sw, fam, rel(swf), low=low, low_source=", ".join(rel(f) for f in fs))
        other = "all" if lowlam_shards == "paper" else "paper"
        S2 = Stats(S.B, S.seed, S.quote)
        sweep_section(S2, "x", "sw", sw, fam, "", low=lows[other][0])
        a = {d["key"]: d for d in S.rows if d["prefix"] == "sw" and d["key"].startswith(("ratio.", "ever."))}
        diffs = [(k, S.text(a[k]), S2.text(d)) for d in S2.rows for k in [d["key"]] if k in a and
                 S.text(a[k]) != S2.text(d)]
        S.note("062 sweep: the budget grid read with %s low-penalty shard files (%d); with %s (%d files) %s" % (
            lowlam_shards, len(fs), other, len(lows[other][1]),
            "the budget numbers are unchanged" if not diffs else "these change: " + "; ".join(
                "%s %s -> %s" % x for x in diffs)))
    ddf = Path(e062) / "out_rescore_dedup" / "eval_dedup.jsonl"
    if ddf.exists():
        dd = clean(read_jsonl([ddf])).drop_duplicates("seed", keep="last")
        dd = dd[~vacuous(dd)].copy()
        dd["pass"] = passes(dd)
        j = dd.set_index("seed")[["pass"]].join(fitted.rename("run"), how="inner")
        S.diff("062 exact-duplicate re-score (rescore_dedup; local GPU vs H100 run: hardware-confounded)", "dedup",
               "effect", "dedup re-score minus the run's own row, paired (leaky targets)", j.run, j["pass"].astype(float),
               src_of([ddf]) + " vs " + src_of(files), macro=True)
    return strong


def run_rebuilt(S, work_tl, rs_v2, rs_asrun, asrun_targets, nulls_dirs, dan159):
    fam = family("turingllm")
    fitted = None
    if rs_v2 and Path(rs_v2, "main").exists():
        fs = row_files(Path(rs_v2) / "main", "main")
        cls = rescore_classify(fs, ("seed", "held"))
        if len(cls):
            sec = "TuringLLM re-scored on the rebuilt held-out sets (086 ctx_v2; headline = >= %d clean rows)" % FLOOR
            head, worst, nb = rebuilt_frame(cls, "strong", fam)
            src = src_of(fs, "held = strong; 086 passrates.classify; below floor %d" % nb)
            pass_breakdown(S, sec, "rb", head, fam, src, worst=worst)
            mh, mw, mb = rebuilt_frame(cls, "mid", fam)
            S.prop(sec, "rb", "pass.midHeld", "held-out mid-band, headline", mh["pass"], src_of(fs, "held = mid"))
            S.prop(sec, "rb", "pass.midHeld.worst", "held-out mid-band, worst case", mw["pass"], src_of(fs, "held = mid"))
            fitted = head.set_index("seed")["pass"].astype(float)
            spf = row_files(Path(rs_v2) / "spec", "spec")
            if spf:
                amplifier_section(S, "TuringLLM specificity on the rebuilt sets (headline targets)", "rb", head,
                                  amp_flags(read_jsonl(spf)), src_of(spf))
            a1f = row_files(Path(rs_v2) / "alpha1", "alpha1")
            if a1f:
                a1c = rescore_classify(a1f)
                if len(a1c):
                    a1h = a1c[a1c.scored & ~a1c.below & ~a1c.vac & ~a1c.rejected]
                    alpha1_section(S, "TuringLLM alpha = 1 on the rebuilt sets", "rb", tag(a1h[["seed", "pass"]], fam),
                                   fitted, fam, src_of(a1f))
                else:
                    S.note("rebuilt alpha = 1: %s has no rows" % src_of(a1f))
    ref = row_files(Path(dan159) / "ref", "rows") if dan159 else []
    swf = row_files(Path(rs_v2) / "sweep", "sweep") if rs_v2 else []
    if ref or swf:
        use = ref if ref else swf
        sw = sweep_from_rows(read_jsonl(use), fam)
        if len(sw) and sw.weighted.any() and (~sw.weighted).any():
            sweep_section(S, "TuringLLM sweep on the rebuilt sets (%s)" % ("083 ref re-score, 13 arms" if ref else
                                                                          "086 sweep re-score, 10 arms"), "rbsw",
                          sw, fam, src_of(use))
    if rs_v2 and rs_asrun and Path(rs_asrun, "main").exists():
        fa, fb = row_files(Path(rs_asrun) / "main", "main"), row_files(Path(rs_v2) / "main", "main")
        ca, cb = rescore_classify(fa, ("seed", "held")), rescore_classify(fb, ("seed", "held"))
        sub = set(Path(asrun_targets).read_text().split()) if asrun_targets and Path(asrun_targets).exists() else None
        if sub is not None and len(ca) and not (sub & set(ca.seed)):
            S.note("leakage: none of the as-run re-score's %d targets is in %s (a smoke run); pairing every target in "
                   "both re-scores instead" % (ca.seed.nunique(), rel(asrun_targets)))
            sub = None
        elif sub is None:
            S.note("leakage: no as-run target list (%s); pairing every target in both re-scores" % asrun_targets)
        elif len(ca):
            S.note("leakage: the frozen subset %s has %d targets; the as-run re-score has %d of them" % (
                rel(asrun_targets), len(sub), len(sub & set(ca.seed))))
        leakage_section(S, "Held-out leakage, paired on the as-run subset (rebuilt minus as run)", "leak",
                        ca, cb, sub, fam,
                        src_of(fa) + " vs " + src_of(fb) + ("; subset %s" % rel(asrun_targets) if sub else ""))
    for lab, d in nulls_dirs:
        if Path(d).exists():
            nulls084_section(S, "084 harder nulls, %s (held-out strongest, draw 0)" % lab, "null" + lab, d,
                             family(lab if lab in ("pythia-70m", "gpt2-small") else "turingllm"),
                             src_of(row_files(d, "nulls"), "split = held, draw 0, vacuous removed"))
    if dan159 and Path(dan159, "table2_rows.csv").exists():
        dan159_section(S, "DAN-159 method comparison (083 Table 2 rows)", "cmp", Path(dan159) / "table2_rows.csv",
                       fam, rel(Path(dan159) / "table2_rows.csv"))


def run_public(S, fam_name, out):
    fam = family(fam_name)
    pre = {"pythia-70m": "pythia", "gpt2-small": "gpttwo"}.get(fam_name, macro_name(fam_name))
    strong, mid, nvac, files = driver_main(out, fam)
    if strong is None:
        S.note("%s: no main eval rows under %s" % (fam_name, rel(out)))
        return
    sec = "%s public run (082)" % fam_name
    src = src_of(files, "held = strong, vacuous removed (%d)" % nvac)
    pass_breakdown(S, sec, pre, strong, fam, src)
    if mid is not None and len(mid):
        S.prop(sec, pre, "pass.midHeld", "held-out mid-band", mid["pass"], src_of(files, "held = mid"))
    spf = row_files(Path(out) / "main", "spec")
    if spf:
        amplifier_section(S, "%s specificity" % fam_name, pre, strong, amp_flags(read_jsonl(spf)), src_of(spf))
    fitted = strong.set_index("seed")["pass"].astype(float)
    a1f = row_files(Path(out) / "alpha1", "alpha1")
    if a1f:
        a1 = clean(read_jsonl(a1f))
        if len(a1):
            a1 = a1.drop_duplicates("seed", keep="last")
            a1 = a1[~vacuous(a1)].copy()
            a1["pass"] = passes(a1)
            alpha1_section(S, "%s alpha = 1" % fam_name, pre, tag(a1, fam), fitted, fam, src_of(a1f))
    swf = row_files(Path(out) / "sweep", "eval")
    if swf:
        sw = sweep_from_rows(read_jsonl(swf), fam)
        if len(sw) and sw.weighted.any() and (~sw.weighted).any():
            sweep_section(S, "%s sweep: WCM vs unweighted" % fam_name, pre + "sw", sw, fam, src_of(swf))
        elif len(sw):
            S.note("%s sweep: only one method in %s (pilot penalties), no WCM-vs-unweighted claims" % (
                fam_name, src_of(swf)))
    nd = Path(out) / "harder_nulls"
    if nd.exists():
        nulls084_section(S, "084 harder nulls, %s (held-out strongest, draw 0)" % fam_name, pre + "null", nd, fam,
                         src_of(row_files(nd, "nulls"), "split = held, draw 0, vacuous removed"))


# ----------------------------------------------------------------------------------------------------- output
def _f(x, dec=1):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else ("%." + str(dec) + "f") % (100 * x)


def to_markdown(S, title):
    L = ["# %s" % title, "",
         "Generated by `experiments/087-intervals/intervals.py` (DAN-163). B = %s target-level bootstrap resamples "
         "(seed %d, per-statistic crc32 offset); quoted interval: %s for single proportions (one-sided exact "
         "Clopper-Pearson upper bound at k = 0), percentile bootstrap for medians, ratios and paired differences. "
         "Pass = DAN-8 (Z, A, C in [0.8, 1.5] and necessity >= 0.9, held-out strongest, activation read; vacuous "
         "denominators removed)." % (thousands(S.B), S.seed, "Wilson" if S.quote == "wilson" else "bootstrap"), ""]
    secs = []
    for d in S.rows:
        if d["section"] not in secs:
            secs.append(d["section"])
    for sec in secs:
        L += ["## " + sec, "", "| statistic | text (quoted) | point | n | k | Wilson 95% | bootstrap 95% | exact CP 95% | "
              "1-sided upper (CP / rule of 3) | macro | source |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in [x for x in S.rows if x["section"] == sec]:
            if d["type"] == "prop":
                cells = [d["label"], S.full(d), _f(d["point"]) + "%", thousands(d["n"]), str(d["k"]),
                         "[%s, %s]" % (_f(d["wilson"][0]), _f(d["wilson"][1])),
                         "[%s, %s]" % (_f(d["boot"][0]), _f(d["boot"][1])),
                         "[%s, %s]" % (_f(d["cp"][0]), _f(d["cp"][1])),
                         ("%s%% / %s%%" % (_f(d["cp_upper"], 2), _f(d["rule3"], 2))) if d["k"] == 0 and d["n"] else
                         ("%s%%" % _f(d["cp_upper"], 2) if d.get("null") and d["n"] else "")]
            elif d["type"] == "diff":
                cells = [d["label"] + " (as %s -> %s; %d up, %d down)" % (_f(d["a"]), _f(d["b"]), d["up"], d["down"]),
                         S.full(d), "%+.2f pp" % (100 * d["point"]) if np.isfinite(d["point"]) else "-",
                         thousands(d["n"]), "", "", "[%+.2f, %+.2f]" % (100 * d["boot"][0], 100 * d["boot"][1])
                         if np.isfinite(d["boot"][0]) else "-", "", ""]
            else:
                cells = [d["label"], S.full(d), ("%.3g" % d["point"]) if np.isfinite(d["point"]) else "-",
                         thousands(d["n"]), "", "", "[%.3g, %.3g]" % d["boot"] if np.isfinite(d["boot"][0]) else "-",
                         "", ""]
            cells += [("`\\%s`" % d["macro"]) if d["macro"] else "", d["source"]]
            L.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
        L.append("")
    if S.notes:
        L += ["## Notes", ""] + ["- " + n for n in S.notes] + [""]
    return "\n".join(L)


def to_tex(S, title):
    L = ["%% %s: generated by experiments/087-intervals/intervals.py (DAN-163); do not edit by hand." % title,
         "%% \\X = value [95%% interval]; \\XN = n; \\XFull = both. Proportions: %s interval (k = 0: one-sided exact "
         "upper bound); medians, ratios, paired differences: target bootstrap, B = %s." % (
             "Wilson" if S.quote == "wilson" else "bootstrap", thousands(S.B))]
    sec = None
    for d in S.rows:
        if not d["macro"]:
            continue
        if d["section"] != sec:
            sec = d["section"]
            L += ["", "%% == %s" % sec]
        L.append("%% %s | %s" % (d["label"], d["source"]))
        L.append("\\newcommand{\\%s}{%s}" % (d["macro"], S.text(d, tex=True)))
        L.append("\\newcommand{\\%sN}{%s}" % (d["macro"], thousands(d["n"]).replace(",", "{,}")))
        L.append("\\newcommand{\\%sFull}{%s}" % (d["macro"], S.full(d, tex=True)))
    return "\n".join(L) + "\n"


def to_json(S):
    def clean_v(v):
        if isinstance(v, (tuple, list)):
            return [clean_v(x) for x in v]
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        return v
    return json.dumps(dict(B=S.B, seed=S.seed, quote=S.quote, notes=S.notes,
                           stats=[{k: clean_v(v) for k, v in d.items()} | {"text": S.full(d)} for d in S.rows]),
                      indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run062", default=str(E062 / "out_full" / "out"))
    ap.add_argument("--e062", default=str(E062))
    ap.add_argument("--no-062", action="store_true", help="skip today's 062 outputs")
    ap.add_argument("--work", help="the 082 chain WORK dir (out_turingllm/, out_<family>/)")
    ap.add_argument("--rescore-v2")
    ap.add_argument("--rescore-asrun")
    ap.add_argument("--asrun-targets", default=str(E086 / "targets_asrun1000.txt"))
    ap.add_argument("--nulls084", action="append", default=[], help="LABEL=DIR (LABEL turingllm | pythia-70m | ...)")
    ap.add_argument("--dan159")
    ap.add_argument("--public", action="append", default=[], help="FAMILY=DIR (a 062-driver OUT dir)")
    ap.add_argument("--lowlam-shards", choices=("paper", "all"), default="paper",
                    help="062 budget grid: paper = eval.shard0.jsonl only (as paper_main.figure_budget); all = every shard")
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--quote", choices=("wilson", "bootstrap"), default="wilson")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    S = Stats(a.B, a.seed, a.quote)
    if not a.no_062:
        run_062(S, a.run062, a.e062, a.lowlam_shards)
    rs_v2, rs_asrun, dan159, nulls = a.rescore_v2, a.rescore_asrun, a.dan159, [x.split("=", 1) for x in a.nulls084]
    public = [x.split("=", 1) for x in a.public]
    if a.work:
        tl = Path(a.work) / "out_turingllm"
        rs_v2 = rs_v2 or (str(tl / "rescore_ctx_v2") if (tl / "rescore_ctx_v2").exists() else None)
        rs_asrun = rs_asrun or (str(tl / "rescore_ctx_asrun1000") if (tl / "rescore_ctx_asrun1000").exists() else None)
        dan159 = dan159 or (str(tl / "dan159") if (tl / "dan159").exists() else None)
        if (tl / "harder_nulls").exists() and not any(l == "turingllm" for l, _ in nulls):
            nulls.append(("turingllm", str(tl / "harder_nulls")))
        for f in ("pythia-70m", "gpt2-small"):
            if (Path(a.work) / ("out_" + f) / "main").exists() and not any(x == f for x, _ in public):
                public.append((f, str(Path(a.work) / ("out_" + f))))
    if rs_v2 or rs_asrun or nulls or dan159:
        run_rebuilt(S, None, rs_v2, rs_asrun, a.asrun_targets, nulls, dan159)
    for f, d in public:
        run_public(S, f, d)
    name = a.name or ("today" if not a.work else "work")
    out = Path(a.out or (HERE / "out" / name))
    out.mkdir(parents=True, exist_ok=True)
    title = "Intervals (%s)" % name
    md = to_markdown(S, title)
    (out / "intervals.md").write_text(md, encoding="utf-8")
    (out / "intervals.tex").write_text(to_tex(S, title), encoding="utf-8")
    (out / "intervals.json").write_text(to_json(S), encoding="utf-8")
    for d in S.rows:
        if d["macro"] or d["type"] != "prop":
            print("%-70s %s" % (d["label"][:70], S.full(d)))
    for n in S.notes:
        print("note:", n)
    print("written:", out / "intervals.md", out / "intervals.tex", out / "intervals.json")
    return S


if __name__ == "__main__":
    main()
