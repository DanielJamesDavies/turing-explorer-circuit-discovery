"""Arena A, A13 (2026-11-05): the tuning sweep's summary and the PRE-REGISTERED selection rule (TUNING.md section 3).
CPU only, no model. Reads, per config of tune_grid.tsv, OUT/<config>/score/rows.jsonl (score_arena_a.py rows, method
wcm) and OUT/<config>/KILLED; writes OUT/summary.md, OUT/summary.json and OUT/CHOSEN.env (the chosen config's env,
for run_pod.sh P5).

Rule (fixed before any run):
  units      the dev targets (tune_dev_targets.txt), held-out strongest ("ho", 16 rows each)
  eligible   a config is eligible when every dev target is either scored or listed in KILLED (a killed / crashed
             fit counts as a FAIL); a config with any target neither scored nor killed is INCOMPLETE and not ranked
  score      P_b = # dev targets with pass_b (Z, A, C in [0.75, 1.5] and nec_b >= 0.9) on ho; a vacuous target = fail
  ranking    (1) max P_b; (2) max P_a (same with nec_a); (3) min median n_latent over the scored dev targets;
             (4) fewer knobs changed vs v4 (base = 0: v4 wins every full tie); (5) grid order
  chosen     the first config in that order. Nothing else (home-turf counts, R, sizes per target) enters the choice.

  .venv/bin/python -B experiments/107-true-baselines/arena_a/tune_summary.py --tune <OUT> --grid <tsv> --dev k1,k2,...
"""
import argparse
import json
import os
import statistics as st
from pathlib import Path


def grid_rows(path):
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        out.append(dict(name=f[0], env=f[1], knobs=int(f[2]), est_gpu_h=float(f[3]), note=f[4] if len(f) > 4 else ""))
    return out


def load_rows(path):
    rows = {}
    if not path.exists():
        return rows
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue                                   # a torn last line (the scorer appends + flushes)
        if r.get("method") == "wcm":
            rows[(r["key"], r["eval_set"])] = r       # last row wins
    return rows


def killed(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                f = line.split("\t")
                out[f[0]] = f[1] if len(f) > 1 else "?"
    return out


def evaluate(cfg, tune, dev):
    rows = load_rows(tune / cfg["name"] / "score" / "rows.jsonl")
    kl = killed(tune / cfg["name"] / "KILLED")
    per, missing = {}, []
    for k in dev:
        ho, home = rows.get((k, "ho")), rows.get((k, "home"))
        if ho is None:
            if k in kl:
                per[k] = dict(status="killed: " + kl[k], pass_b=False, pass_a=False)
            else:
                missing.append(k)
            continue
        per[k] = dict(status="vacuous" if ho.get("vacuous") else "scored", n=ho.get("n_latent"),
                      Z=ho.get("Z"), A=ho.get("A"), C=ho.get("C"), nec_a=ho.get("nec_a"), nec_b=ho.get("nec_b"),
                      nec_b_tc=ho.get("nec_b_tc"), pass_a=bool(ho.get("pass_a")), pass_b=bool(ho.get("pass_b")),
                      home_pass_b=None if home is None else home.get("pass_b"),
                      home_pass_a=None if home is None else home.get("pass_a"))
    ns = [v["n"] for v in per.values() if v.get("n") is not None]
    return dict(cfg, per=per, missing=missing, complete=not missing,
                P_b=sum(v["pass_b"] for v in per.values()), P_a=sum(v["pass_a"] for v in per.values()),
                home_P_b=sum(bool(v.get("home_pass_b")) for v in per.values()),
                median_n=st.median(ns) if ns else None, n_killed=sum(v["status"].startswith("killed") for v in per.values()))


def f2(x):
    return "-" if x is None else ("%.2f" % x if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--tune", default=str(here / "out" / "tune"))
    ap.add_argument("--grid", default=str(here / "tune_grid.tsv"))
    ap.add_argument("--dev", default="")
    a = ap.parse_args()
    tune = Path(a.tune)
    dev = [k for k in (a.dev.split(",") if a.dev else (here / "tune_dev_targets.txt").read_text().split()) if k]
    grid = grid_rows(a.grid)
    res = [evaluate(c, tune, dev) for i, c in enumerate(grid)]
    for i, r in enumerate(res):
        r["order"] = i
    ranked = sorted([r for r in res if r["complete"]],
                    key=lambda r: (-r["P_b"], -r["P_a"], r["median_n"] if r["median_n"] is not None else float("inf"),
                                   r["knobs"], r["order"]))
    chosen = ranked[0] if ranked else None
    L = ["# Arena A WCM tuning (A13): summary", "",
         "Dev set: %d targets (%s). Held-out strongest (ho). Rule (pre-registered, TUNING.md section 3): max P_b, then "
         "max P_a, then min median n, then fewer changed knobs (v4 = 0), then grid order. Killed fits = fail; "
         "incomplete configs are not ranked." % (len(dev), ", ".join(dev)), "",
         "| rank | config | env | complete | P_b ho | P_a ho | P_b home | median n | killed | est GPU-h |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    rank_of = {r["name"]: i + 1 for i, r in enumerate(ranked)}
    for r in sorted(res, key=lambda r: (rank_of.get(r["name"], 99), r["order"])):
        L.append("| %s | %s%s | `%s` | %s | %d | %d | %d | %s | %d | %.2f |" % (
            rank_of.get(r["name"], "-"), r["name"], " **(chosen)**" if chosen and r["name"] == chosen["name"] else "",
            r["env"], "yes" if r["complete"] else "no (%d missing)" % len(r["missing"]), r["P_b"], r["P_a"],
            r["home_P_b"], f2(r["median_n"]), r["n_killed"], r["est_gpu_h"]))
    L += ["", "## Per target (ho): Z / A / C / nec a / nec b, pass b", "",
          "| target | " + " | ".join(r["name"] for r in res) + " |", "|---|" + "---|" * len(res)]
    for k in dev:
        cells = []
        for r in res:
            v = r["per"].get(k)
            if v is None:
                cells.append("–")
            elif v["status"] != "scored":
                cells.append(v["status"])
            else:
                cells.append("%s / %s / %s / %s / %s, n %s %s" % (f2(v["Z"]), f2(v["A"]), f2(v["C"]), f2(v["nec_a"]),
                                                                  f2(v["nec_b"]), v["n"], "**PASS**" if v["pass_b"] else "fail"))
        L.append("| %s | %s |" % (k, " | ".join(cells)))
    L += ["", "Chosen: **%s**" % (chosen["name"] if chosen else "none (no complete config)")]
    tmp = tune / "summary.md.tmp"
    tune.mkdir(parents=True, exist_ok=True)
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, tune / "summary.md")
    (tune / "summary.json").write_text(json.dumps(dict(dev=dev, configs=res, ranked=[r["name"] for r in ranked],
                                                       chosen=chosen["name"] if chosen else None), indent=1) + "\n")
    if chosen:
        env = [] if chosen["env"] in ("REUSE", "-") else chosen["env"].split()
        (tune / "CHOSEN.env").write_text(
            "# arena A WCM tuning (A13) choice, written by tune_summary.py; on top of v4 for run_pod.sh P5 (LAM = the "
            "pod's curve)\nAA_TUNE_CONFIG=%s\n%s" % (chosen["name"], "".join(e + "\n" for e in env)))
    print("\n".join(L))


if __name__ == "__main__":
    main()
