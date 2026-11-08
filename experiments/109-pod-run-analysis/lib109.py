"""Shared pieces of the 109 pod-run analysis (2026-11-07/08 production run, WCM v4 on three models).

Every number in every 109 table comes from a script that reads the unpacked result archives under ARCHIVES
(default experiments/109-pod-run-analysis/out/archives, one folder per tgz, made by unpack.sh). Nothing is typed in.

Not re-implemented here (imported):
  - the pass rule and the vacuous exclusion: experiments/062-h100-protocol-v1/merge.py `passes`, `vacuous`, `HEAD`,
    `BAND` (0.75, 1.5), `NEC_MIN` (0.9). Z = free0_tk, A = freeM_topk_tk, C = freeN_topk_tk, necessity =
    phi_sup_blind_tk; sufficiency to induce = phi_cf_alpha_blind_tk (reported, not in the rule).
  - depth bands per family: experiments/062-h100-protocol-v1/family_table.py (TuringLLM and GPT-2 L0-4 / L5-7 / L8-11,
    Pythia L0-1 / L2-3 / L4-5).
  - LaTeX number helpers: paper/tables/common.py (tex_int, tex_num, DASH).

Row conventions (as merge.py / paper/tables): drop rows with an "error" (or "skip") key, keep the last row per
(seed, held[, arm]); eval rows have held in {"strong", "mid"}; held-out strongest is primary, mid-band reported.
"""
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)   # pandas fillna downcasting notices from merge.vacuous

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E062 = ROOT / "experiments" / "062-h100-protocol-v1"
for _p in (str(ROOT / "src"), str(E062)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import merge as M  # noqa: E402  (the DAN-8 rule, single source; its depth_band is TuringLLM's, not used here)
from family_table import band_label, family  # noqa: E402


def _paper_tables_common():
    import importlib.util
    spec = importlib.util.spec_from_file_location("paper_tables_common", ROOT / "paper" / "tables" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PTC = _paper_tables_common()
DASH, tex_int, tex_num = _PTC.DASH, _PTC.tex_int, _PTC.tex_num

ARCHIVES = Path(os.environ.get("ARCHIVES", str(HERE / "out" / "archives")))
OUT = Path(os.environ.get("OUT109", str(HERE / "out")))
TABLES = Path(os.environ.get("TABLES109", str(ROOT / "paper" / "tables")))
FIGURES = Path(os.environ.get("FIGURES109", str(ROOT / "paper" / "figures")))
PREFIX = "run2026_"
BOOT_B = int(os.environ.get("BOOT_B", "4000"))
BOOT_SEED = 20261108

HEAD, BAND, NEC_MIN = M.HEAD, M.BAND, M.NEC_MIN
NEC, IND = "phi_sup_blind_tk", "phi_cf_alpha_blind_tk"
passes, vacuous = M.passes, M.vacuous

MODELS = ["turingllm", "pythia-70m", "gpt2-small"]
LABEL = {"turingllm": "TuringLLM", "pythia-70m": "Pythia-70m", "gpt2-small": "GPT-2 small"}
KIND_ORDER = ["resid", "resid_mid", "mlp", "attn"]
KIND_LABEL = {"resid": "residual", "resid_mid": "residual (mid)", "mlp": "MLP", "attn": "attention"}

# Where each model's outputs sit inside an unpacked archive (relative to the archive folder). Public models: the
# 082 chain's out_<model>; TuringLLM: the T3+ OUT (094/out_v2), the track's work folder (082/out_turingllm: lists,
# records, harder nulls) and the T10 without-R OUT (094/out_v2_noR).
_TL = "root/turing-tl/experiments"
LOCS = {
    "pythia-70m": {"out": "experiments/082-public-runs/out_pythia-70m", "run": "experiments/082-public-runs",
                   "stores": "outputs_public/pythia-70m"},
    "gpt2-small": {"out": "experiments/082-public-runs/out_gpt2-small", "run": "experiments/082-public-runs",
                   "stores": "outputs_public/gpt2-small"},
    "turingllm": {"out": _TL + "/094-tl-v2-records/out_v2", "work": _TL + "/082-public-runs/out_turingllm",
                  "nor": _TL + "/094-tl-v2-records/out_v2_noR"},
}
ARCHIVE_GLOB = {"pythia-70m": "pub-pythia-70m-*", "gpt2-small": "pub-gpt2-small-*", "turingllm": "pub-turingllm-*"}


def find(model, where, sub=""):
    """Existing directories <archive>/<LOCS[model][where]>/<sub> over every unpacked archive of the model, sorted by
    archive name (so a later stage's archive is read after an earlier one and wins the keep-last)."""
    rel = LOCS[model].get(where)
    if rel is None:
        return []
    out = []
    for a in sorted(ARCHIVES.glob(ARCHIVE_GLOB[model])):
        if a.is_dir() and not a.name.endswith("-stores"):
            d = a / rel / sub if sub else a / rel
            if d.exists():
                out.append(d)
    return out


def find_file(model, where, rel):
    for d in find(model, where):
        p = d / rel
        if p.exists():
            return p
    return None


def read_rows(dirs, stem):
    """Every JSON row of <dir>/<stem>.jsonl and <dir>/<stem>.shard*.jsonl over dirs, in order (bad lines skipped)."""
    rows = []
    for d in dirs:
        d = Path(d)
        for p in [d / ("%s.jsonl" % stem)] + sorted(d.glob("%s.shard*.jsonl" % stem)):
            if p.exists():
                with open(p, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            rows.append(json.loads(line))
                        except ValueError:
                            continue
    return rows


def frame(rows, keys=("seed", "held")):
    """Rows without an error / skip key as a DataFrame, last row per keys (keys missing from every row are ignored)."""
    good = [r for r in rows if "error" not in r and not r.get("skip")]
    df = pd.DataFrame(good)
    if df.empty:
        return df
    ks = [k for k in keys if k in df]
    if ks:
        df = df.drop_duplicates(ks, keep="last")
    return df.reset_index(drop=True)


def bands_of(model):
    fam = family(model)
    return [(band_label(lo, hi), lo, hi) for lo, hi in fam["bands"]]


def band_of(model, layer):
    for name, lo, hi in bands_of(model):
        if lo <= int(layer) <= hi:
            return name
    return bands_of(model)[-1][0]


def add_site(df, model):
    df = df.copy()
    df["layer"] = df.seed.str.split(".").str[0].astype(int)
    df["kind"] = df.seed.str.split(".").str[1]
    df["band"] = df.layer.map(lambda l: band_of(model, l))
    return df


def score(df):
    """Adds vac (merge.vacuous) and pass (merge.passes on the non-vacuous rows; False on vacuous rows)."""
    df = df.copy()
    for c in HEAD + [NEC, IND, "n"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["vac"] = vacuous(df).values
    df["pass"] = False
    ok = ~df.vac
    if ok.any():
        df.loc[ok, "pass"] = passes(df[ok]).values
    df["pass"] = df["pass"].astype(bool)
    return df


def main_eval(model):
    """The main run's scored eval rows (both helds), with site columns, vac and pass. Empty if absent."""
    df = frame(read_rows(find(model, "out", "main"), "eval"), keys=("seed", "held"))
    if df.empty:
        return df
    return score(add_site(df, model))


def boot_ci(x, B=None, seed=BOOT_SEED, stat=np.mean):
    """95% percentile bootstrap interval of stat over targets (resampling targets with replacement)."""
    x = np.asarray(x, dtype=float)
    x = np.sort(x[~np.isnan(x)])                          # sorted: the interval does not depend on row order
    if len(x) == 0:
        return float("nan"), float("nan")
    B = B or BOOT_B
    rng = np.random.default_rng(seed)
    if stat is np.mean:
        idx = rng.integers(0, len(x), size=(B, len(x)))
        s = x[idx].mean(axis=1)
    else:
        s = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def boot_paired_diff(a, b, B=None, seed=BOOT_SEED):
    """95% bootstrap interval of mean(a) - mean(b) over paired targets."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(B or BOOT_B, len(a)))
    s = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def pass_cell(p, n_ok=None):
    """'81.9 [78.5, 85.2]' from a boolean series (pass over non-vacuous rows)."""
    p = pd.Series(p).astype(float)
    if len(p) == 0:
        return dict(k=0, n=0, pct=float("nan"), lo=float("nan"), hi=float("nan"))
    lo, hi = boot_ci(p.values)
    return dict(k=int(p.sum()), n=int(len(p)), pct=100 * p.mean(), lo=100 * lo, hi=100 * hi)


def fmt_pass(c, tex=False):
    if not c or c["n"] == 0 or math.isnan(c["pct"]):
        return DASH if tex else "-"
    s = "%.1f [%.1f, %.1f]" % (c["pct"], c["lo"], c["hi"])
    return s.replace("[", "{[}").replace("]", "{]}") if tex else s


def med_iqr(s, p=0, tex=False):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if s.empty:
        return DASH if tex else "-"
    f = "%%.%df" % p
    out = (f + " (" + f + "--" + f + ")") % (s.median(), s.quantile(.25), s.quantile(.75))
    return out if tex else out.replace("--", "-")


def tex_escape(s):
    return (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("<=", r"$\le$")
            .replace(">=", r"$\ge$"))


def write_table(name, df, columns, align, note=None, csv_df=None, footer=None):
    """paper/tables/run2026_<name>.tex (a booktabs tabular, no float, for \\input inside the owner's table) and
    out/run2026_<name>.csv (csv_df, or df). columns: header cells (LaTeX); df: rows of already-formatted cells.
    A row whose first cell starts with '\\midrule' is written as a rule. footer: visible note lines (LaTeX) set
    under the bottom rule, each spanning every column in footnote size."""
    TABLES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["%% generated by experiments/109-pod-run-analysis (%s); do not edit by hand" % name,
             r"\begin{tabular}{%s}" % align, r"\toprule", " & ".join(columns) + r" \\", r"\midrule"]
    for row in df:
        if row and str(row[0]).startswith(r"\midrule"):
            lines.append(r"\midrule")
            continue
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines.append(r"\bottomrule")
    for f in footer or []:
        lines.append(r"\multicolumn{%d}{l}{\footnotesize %s} \\" % (len(columns), f))
    lines.append(r"\end{tabular}")
    if note:
        lines.append("%% " + note.replace("\n", "\n%% "))
    (TABLES / ("%s%s.tex" % (PREFIX, name))).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if csv_df is not None:
        csv_df.to_csv(OUT / ("%s%s.csv" % (PREFIX, name)), index=False)
    return TABLES / ("%s%s.tex" % (PREFIX, name))


def write_json(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / ("%s%s.json" % (PREFIX, name))
    p.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    return p


def not_available(what):
    print("[109] %s: not yet available" % what)


def targets_list(model, which="main"):
    """The frozen target list of the run (public: 082 targets_<model>_<which>.txt; TuringLLM: lists/targets_tl8k*.txt
    with which in main | sweep190 | nulls126 | stable128 | case6 | gallery9 | gate64)."""
    if model == "turingllm":
        name = "targets_tl8k.txt" if which == "main" else "targets_tl8k_%s.txt" % which
        p = find_file(model, "work", "lists/" + name)
    else:
        p = find_file(model, "run", "targets_%s_%s.txt" % (model, which))
    if p is None:
        return None
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]


def save_with_legend(fig, path, top):
    """style.save_figure, but the tight layout leaves the band above `top` (figure fraction) for a figure-level legend
    (save_figure's own tight_layout would pull the axes over it). Writes the PNG and the PDF."""
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def figure_style():
    from analysis import style
    plt = style.configure_matplotlib()
    return plt, style
