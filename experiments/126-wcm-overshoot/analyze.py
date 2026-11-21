"""Exp 126 (CPU): the overshoot tables from score_arena_a rows.

Input: score_arena_a rows.jsonl files (one row per unit x eval set; WCM rows only). Default: every
OUT/score/<setting>/rows.jsonl under this experiment; --rows PATH[=NAME] adds files (NAME defaults to the parent dir).
A setting = (file, WCM label): named after the file when the label is BASE_LABEL (l1e-3), else "<name>:<label>"
(arm D: "s400:c2"). The last row per (uid, eval set) wins (resumed runs append).

Per setting and target: ho and home Z / A / C / nec_b / pass_b, n (latents), the ho_xdoc subset (ho windows from
documents in none of the training windows: n and max(Z, A, C)), the circuit's alpha p90 / max (--alphas: reads the
circuit file, CPU), and
  overshoot   max(Z, A, C) on ho > 1.5 while home passes (pass_b True): fits its own contexts, overshoots held-out
  over_ho     max(Z, A, C) on ho > 1.5 (whatever home does)
Tables (markdown, --out [OUT/report.md]): settings summary, delta vs --baseline on the shared targets, a target x
setting pivot of ho max(Z, A, C), and per-setting detail.

  WSL, repo root:
  ./.venv/bin/python -B experiments/126-wcm-overshoot/analyze.py [--rows FILE[=NAME] ...] [--no-default] \
      [--baseline s400] [--alphas] [--out experiments/126-wcm-overshoot/out/report.md]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A = ROOT / "experiments" / "107-true-baselines" / "arena_a"
BAND, NEC_MIN, OVER = (0.75, 1.5), 0.9, 1.5
BASE_LABEL = "l1e-3"
# the order settings appear in (others follow, sorted)
ORDER = ["s100", "s200", "s400", "s800", "n8", "n16", "s400:c1.5", "s400:c2", "s400:c3", "owt131k"]


def jl(path):
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def mx(r):
    v = [r.get(f) for f in ("Z", "A", "C")]
    return None if any(x is None for x in v) else max(v)


def f2(v):
    return "-" if v is None else ("%.2f" % v if isinstance(v, float) else str(v))


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def tkey(k):
    return tuple(int(x) if x.isdigit() else x for x in k.split("."))


def load(files):
    """{setting: {target: {"ho": row, "home": row}}}"""
    S = {}
    for path, name in files:
        last = {}
        for r in jl(path):
            if r.get("method") != "wcm":
                continue
            last[(r["uid"], r["eval_set"])] = r
        for (uid, es), r in last.items():
            lab = r.get("label") or ""
            s = name if lab == BASE_LABEL else "%s:%s" % (name, lab)
            S.setdefault(s, {}).setdefault(r["key"], {})[es] = r
    return S


def alpha_stats(paths):
    """{circuit path: (p90, max)} (CPU; registers the family so the circuits unpickle)"""
    for p in (str(ROOT / "src"), str(A)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import wcm_tcz as WT
    WT.register()
    from eval.wcm_eval import circuit_alphas
    out = {}
    for p in sorted(set(paths)):
        q = Path(p) if Path(p).is_absolute() else ROOT / p
        if not q.exists():
            continue
        v = sorted(x for d in circuit_alphas(torch.load(q, weights_only=False, map_location="cpu")).values()
                   for x in d.values())
        if v:
            out[p] = (v[min(len(v) - 1, int(0.9 * len(v)))], v[-1])
    return out


def entry(t, al):
    ho, hm = t.get("ho"), t.get("home")
    e = dict(n=(ho or hm or {}).get("n_latent"), ho=ho, home=hm, ho_max=mx(ho) if ho else None,
             home_max=mx(hm) if hm else None, ho_pass=None if not ho else ho.get("pass_b"),
             home_pass=None if not hm else hm.get("pass_b"))
    e["over_ho"] = None if e["ho_max"] is None else e["ho_max"] > OVER
    e["overshoot"] = None if e["ho_max"] is None or hm is None else bool(e["over_ho"] and e["home_pass"] is True)
    x = (ho or {}).get("xdoc") or {}
    e["xdoc_n"] = x.get("n")                                # 0 = every ho window shares a training document
    e["xdoc_max"] = mx(x) if x.get("idx") else None
    src = (ho or hm or {}).get("src")
    e["alpha"] = al.get(src) if src else None
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", action="append", default=[], help="rows.jsonl[=NAME] (repeatable)")
    ap.add_argument("--no-default", action="store_true", help="do not scan OUT/score/*/rows.jsonl")
    ap.add_argument("--baseline", default="s400")
    ap.add_argument("--alphas", action="store_true", help="read each circuit's alpha p90 / max (CPU, ~30 s)")
    ap.add_argument("--out", default=str(HERE / "out" / "report.md"))
    ap.add_argument("--title", default="Exp 126: WCM overshoot on shallow tc targets (scope a2, lambda 1e-3)")
    a = ap.parse_args()
    files = []
    if not a.no_default:
        files += [(p, p.parent.name) for p in sorted((HERE / "out" / "score").glob("*/rows.jsonl"))]
    for x in a.rows:
        p, _, n = x.partition("=")
        files.append((Path(p), n or Path(p).parent.name))
    if not files:
        raise SystemExit("no rows files (OUT/score/*/rows.jsonl or --rows)")
    S = load(files)
    al = alpha_stats([r.get("src") for s in S.values() for t in s.values() for r in t.values()]) if a.alphas else {}
    E = {s: {k: entry(t, al) for k, t in ts.items()} for s, ts in S.items()}
    names = [s for s in ORDER if s in E] + sorted(s for s in E if s not in ORDER)

    L = ["# %s" % a.title, "",
         "Inputs: %s." % ", ".join("`%s` (%s)" % (p, n) for p, n in files),
         "Pass rule (pass_b): Z, A, C in [%.2f, %.2f] and nec_b >= %.1f. **overshoot** = max(Z, A, C) on ho > %.1f "
         "while home passes (pass_b); over_ho = max(Z, A, C) on ho > %.1f. ho = the 16 held-out strongest; home = the "
         "circuit's own K = 8 highest-peak training-strong rows; xdoc = the ho windows from documents in no training "
         "window." % (BAND[0], BAND[1], NEC_MIN, OVER, OVER), "",
         "## Settings", "",
         "| setting | targets | ho pass b | home pass b | overshoot | over_ho | median ho max(Z,A,C) | median home "
         "max(Z,A,C) | median n | median xdoc n | median xdoc max(Z,A,C) | median alpha p90 / max |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in names:
        es = list(E[s].values())
        cnt = lambda f: "%d / %d" % (sum(e[f] is True for e in es), sum(e[f] is not None for e in es))  # noqa: E731
        ap90 = med([e["alpha"][0] for e in es if e["alpha"]])
        amax = med([e["alpha"][1] for e in es if e["alpha"]])
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            s, len(es), cnt("ho_pass"), cnt("home_pass"), cnt("overshoot"), cnt("over_ho"),
            f2(med([e["ho_max"] for e in es])), f2(med([e["home_max"] for e in es])), f2(med([e["n"] for e in es])),
            f2(med([e["xdoc_n"] for e in es])), f2(med([e["xdoc_max"] for e in es])),
            "-" if ap90 is None else "%.2f / %.2f" % (ap90, amax)))
    n_head = len(L)

    if a.baseline in E:
        B = E[a.baseline]
        L += ["", "## Against %s (shared targets)" % a.baseline, "",
              "| setting | shared | median d ho max(Z,A,C) | overshoot base -> setting | fixed (base overshoot, setting "
              "not) | new (setting overshoot, base not) | ho pass b base -> setting |", "|---|---|---|---|---|---|---|"]
        for s in names:
            if s == a.baseline:
                continue
            sh = sorted(set(B) & set(E[s]), key=tkey)
            if not sh:
                continue
            d = [E[s][k]["ho_max"] - B[k]["ho_max"] for k in sh
                 if E[s][k]["ho_max"] is not None and B[k]["ho_max"] is not None]
            ob = sum(B[k]["overshoot"] is True for k in sh)
            os_ = sum(E[s][k]["overshoot"] is True for k in sh)
            fixed = [k for k in sh if B[k]["overshoot"] is True and E[s][k]["overshoot"] is False]
            new = [k for k in sh if B[k]["overshoot"] is False and E[s][k]["overshoot"] is True]
            L.append("| %s | %d | %s | %d -> %d | %s | %s | %d -> %d |" % (
                s, len(sh), "-" if not d else "%+.2f" % st.median(d), ob, os_, " ".join(fixed) or "-",
                " ".join(new) or "-", sum(B[k]["ho_pass"] is True for k in sh), sum(E[s][k]["ho_pass"] is True for k in sh)))

    tg = sorted({k for s in names for k in E[s]}, key=tkey)
    L += ["", "## ho max(Z, A, C) per target (O = overshoot, P = ho pass b, h = home fails)", "",
          "| target | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in tg:
        cells = []
        for s in names:
            e = E[s].get(k)
            if not e or e["ho_max"] is None:
                cells.append("")
                continue
            tag = ("O" if e["overshoot"] else "") + ("P" if e["ho_pass"] else "") + ("h" if e["home_pass"] is False else "")
            cells.append("%.2f %s" % (e["ho_max"], tag))
        L.append("| %s | %s |" % (k, " | ".join(cells)))

    L += ["", "## Detail", ""]
    for s in names:
        L += ["### %s" % s, "",
              "| target | n | ho Z / A / C / nec_b | ho pass | home Z / A / C / nec_b | home pass | xdoc n / max(Z,A,C) | "
              "alpha p90 / max | overshoot |", "|---|---|---|---|---|---|---|---|---|"]
        for k in sorted(E[s], key=tkey):
            e = E[s][k]
            zacn = lambda r: "-" if not r else " / ".join(f2(r.get(f)) for f in ("Z", "A", "C", "nec_b"))  # noqa: E731
            L.append("| %s | %s | %s | %s | %s | %s | %s / %s | %s | %s |" % (
                k, f2(e["n"]), zacn(e["ho"]), f2(e["ho_pass"]), zacn(e["home"]), f2(e["home_pass"]), f2(e["xdoc_n"]),
                f2(e["xdoc_max"]), "-" if not e["alpha"] else "%.2f / %.2f" % e["alpha"], f2(e["overshoot"])))
        L.append("")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(L) + "\n")
    tmp.replace(out)
    js = {s: {k: {f: v for f, v in e.items() if f not in ("ho", "home")} for k, e in E[s].items()} for s in names}
    out.with_suffix(".json").write_text(json.dumps(js, indent=1))
    print("\n".join(L[:n_head]))
    print("report -> %s" % out)


if __name__ == "__main__":
    main()
