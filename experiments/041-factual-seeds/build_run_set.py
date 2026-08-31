"""Merge chosen seeds from the screened pools into ONE run file for the
harness, so a concept run is explicit about which seeds it contains and
where each came from.

  python build_run_set.py 6/455 8/2415 12/13697 12/2064 ...
  python build_run_set.py --file chosen.txt

Pools searched, in order: ww2_seeds.pt, concept_seeds.pt,
factual_seeds.pt. All three were screened on the SAME 20k wikitext
windows with the SAME firing band, and all carry the same token tensor,
so merging is safe; the script asserts the token tensors match rather
than assuming it.

Writes run_seeds.pt (+ run_set.md listing what went in and why).
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
POOLS = ["chem_seeds.pt", "ww2_seeds.pt", "concept_seeds.pt", "factual_seeds.pt"]


def main():
    args = sys.argv[1:]
    if args and args[0] == "--file":
        args = [l.split("#")[0].strip()
                for l in open(args[1]) if l.split("#")[0].strip()]
    if not args:
        print(__doc__); return

    pools, toks = {}, None
    for p in POOLS:
        f = HERE / p
        if not f.exists():
            print("  (pool missing, skipped: %s)" % p); continue
        d = torch.load(f, weights_only=False)
        if toks is None:
            toks = d["tokens"]
        else:
            assert torch.equal(toks, d["tokens"]), \
                "%s was screened on DIFFERENT windows -- refusing to merge" % p
        pools[p] = d["seeds"]

    out, report = {}, ["# Run set", "",
                       "| seed | pool | fire_frac | max_act | label |",
                       "|---|---|---|---|---|"]
    for key in args:
        for p, seeds in pools.items():
            if key in seeds:
                out[key] = seeds[key]
                report.append("| %s | %s | %.4f | %.2f | %s |" % (
                    key, p, seeds[key]["fire_frac"], seeds[key]["max_act"],
                    seeds[key].get("label", "")))
                break
        else:
            print("!! %s not found in any screened pool "
                  "(not in band, or never a candidate)" % key)
    torch.save({"tokens": toks, "seeds": out}, HERE / "run_seeds.pt")
    (HERE / "run_set.md").write_text("\n".join(report), encoding="utf-8",
                                     newline="")
    print("\n".join(report))
    print("\n%d seeds -> run_seeds.pt" % len(out))


if __name__ == "__main__":
    main()
