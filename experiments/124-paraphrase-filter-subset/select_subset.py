"""124 subset (CPU): ~300 TuringLLM production targets, stratified by depth band x site kind, seeded and recorded.

Population: the production held-out-strongest rows (T3 main, through 109 lib109 = 062 merge.py pass and vacuous rules)
that are non-vacuous (the headline population) and have a production circuit file, a T2 record and a T1 scan entry
(what the build needs). Allocation is proportional to the population's cells (largest remainder), so the sample's
pass rate estimates the production rate directly; cells hold 25-45 targets each.

  python -X utf8 experiments/124-paraphrase-filter-subset/select_subset.py
  env: SUBSET_N (300)  SUBSET_SEED (20261116)
Writes targets_subset.txt (sorted keys) and out/subset.json (seed, allocation, population sizes, production pass rate
of the sample and the population per cell).
"""
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import para124 as P  # noqa: E402

C = P.C
sys.path.insert(0, str(C.EXP / "109-pod-run-analysis"))
sys.path.insert(0, str(C.EXP / "086-heldout-rebuild"))
import lib109 as L  # noqa: E402


def population():
    dirs = [d / "main" for d in L.find("turingllm", "out") if (d / "main").exists()]
    df = L.frame(L.read_rows(dirs, "eval"))
    df = L.score(L.add_site(df[df.held == "strong"], "turingllm")).set_index("seed")
    n_all = len(df)
    df = df[~df.vac.astype(bool)]
    tp = C.tl_paths()
    circ = {p.stem for p in (tp["out"] / "main" / "circuits").glob("*.pt")}
    recs = {p.stem for p in tp["ctx"].glob("*.pt")}
    import scan as SC
    where = SC.load_scan(tp["scan"])["where"]
    ok = df[[k in circ and k in recs and k in where for k in df.index]]
    return df, ok, n_all


def main():
    n = int(os.environ.get("SUBSET_N", P.SUBSET_N))
    seed = int(os.environ.get("SUBSET_SEED", P.SUBSET_SEED))
    nonvac, pop, n_all = population()
    keys, alloc = P.stratified_subset(pop, n, seed)
    sel = pop.loc[keys]
    P.subset_path().write_text("\n".join(keys) + "\n")
    cells = {}
    for (b, k), g in pop.groupby(["band", "kind"]):
        s = sel[(sel.band == b) & (sel.kind == k)]
        cells["%s/%s" % (b, k)] = dict(population=int(len(g)), pop_pass_pct=100 * float(g["pass"].mean()),
                                       sample=int(len(s)), sample_pass_pct=100 * float(s["pass"].mean()) if len(s) else None)
    info = dict(seed=seed, n=len(keys), rule="proportional allocation (largest remainder) over band x kind; seeded draw "
                "inside each cell from the sorted keys", production_rows=n_all, non_vacuous=int(len(nonvac)),
                population=int(len(pop)), population_pass_pct=100 * float(pop["pass"].mean()),
                sample_pass_pct=100 * float(sel["pass"].mean()), allocation=alloc, cells=cells,
                keys_md5=hashlib.md5(("\n".join(keys) + "\n").encode()).hexdigest(),
                targets_file=str(P.subset_path().name))
    P.write_json(info, P.WORK / "subset.json")
    print("population %d (of %d non-vacuous, %d rows); picked %d; production pass: population %.1f%%, sample %.1f%%"
          % (len(pop), len(nonvac), n_all, len(keys), info["population_pass_pct"], info["sample_pass_pct"]))
    for c, v in cells.items():
        print("  %-12s pop %4d (%.1f%%)  sample %3d (%.1f%%)" % (c, v["population"], v["pop_pass_pct"], v["sample"],
                                                                 v["sample_pass_pct"] or 0))


if __name__ == "__main__":
    main()
