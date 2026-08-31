"""(2) Do the arms disagree on the METRICS, or on the LATENTS?

The floor panel showed the arms score wildly differently. That is
compatible with two very different worlds:

  MEASUREMENT  they select nearly the same latents, and the metrics
               differ because of weighting/margin/magnitude. Then this
               whole thread is about evaluation convention, and the
               underlying "circuit" is roughly one object.
  MECHANISM    they select different latents. Then each objective is
               finding a genuinely different subgraph and "the circuit
               for seed X" is not well-posed without naming the
               objective.

Reported three ways, because each answers a different question:
  jaccard   |A n B| / |A u B| — symmetric similarity
  contain   |A n B| / min(|A|,|B|) — is the smaller set a SUBSET of the
            larger? (high containment with low jaccard = nesting, i.e.
            the same object at two resolutions)
  matched   both arms cut to the same n by their own ranking, so size
            cannot drive the number

  python experiments/026-floor-isolation/overlap.py
"""
import gzip
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).parent

mem = {}
with gzip.open(HERE / "members.jsonl.gz", "rt") as fh:
    for line in fh:
        if line.strip():
            r = json.loads(line)
            mem[(r["latent"], r["arm"], r["l1"])] = {
                (a, b, c) for a, b, c in r["members"]}
seeds = sorted({k[0] for k in mem})
arms = sorted({k[1] for k in mem})
lams = sorted({k[2] for k in mem})
print("%d circuits | %d seeds | arms %s" % (len(mem), len(seeds), arms))


def jac(a, b):
    return len(a & b) / max(len(a | b), 1)


def contain(a, b):
    return len(a & b) / max(min(len(a), len(b)), 1)


print("\n=== SAME LAMBDA, pairwise (mean over seeds x lambdas) ===")
print("%-24s %-9s %-9s %s" % ("pair", "jaccard", "contain", "n ratio"))
for a1, a2 in itertools.combinations(arms, 2):
    js, cs, rs = [], [], []
    for s in seeds:
        for lam in lams:
            A, B = mem.get((s, a1, lam)), mem.get((s, a2, lam))
            if not A or not B:
                continue
            js.append(jac(A, B)); cs.append(contain(A, B))
            rs.append(max(len(A), len(B)) / max(min(len(A), len(B)), 1))
    if js:
        print("%-24s %-9.3f %-9.3f %.1fx"
              % ("%s vs %s" % (a1, a2), st.mean(js), st.mean(cs), st.mean(rs)))

print("\n=== MATCHED SIZE: each arm cut to the smaller arm's n ===")
print("(both sets truncated by their own ranking is not possible here — "
      "membership is unordered — so this instead compares the two arms'\n"
      " lambda settings whose n are CLOSEST, removing the size confound.)")
print("%-24s %-9s %-9s %s" % ("pair", "jaccard", "contain", "n (a1/a2)"))
for a1, a2 in itertools.combinations(arms, 2):
    js, cs, ns = [], [], []
    for s in seeds:
        best = None
        for l1 in lams:
            for l2 in lams:
                A, B = mem.get((s, a1, l1)), mem.get((s, a2, l2))
                if not A or not B:
                    continue
                d = abs(len(A) - len(B)) / max(len(A), len(B))
                if best is None or d < best[0]:
                    best = (d, A, B)
        if best:
            _, A, B = best
            js.append(jac(A, B)); cs.append(contain(A, B))
            ns.append((len(A), len(B)))
    if js:
        print("%-24s %-9.3f %-9.3f %s"
              % ("%s vs %s" % (a1, a2), st.mean(js), st.mean(cs),
                 "%d/%d" % (int(st.mean(x for x, _ in ns)),
                            int(st.mean(y for _, y in ns)))
                 if ns else "-"))

print("\n=== SIZE-MATCHED RANDOM NULL ===")
print("A random pair of sets of these sizes drawn from the same live pool "
      "would give jaccard ~0; any\nvalue well above 0 is real shared "
      "selection. Reported per pair above — compare to the ~0.004 uniform\n"
      "null measured in 024-l2-crossover.")

print("\n=== per-arm size by lambda (mean over seeds) ===")
print("%-12s %s" % ("arm", "  ".join("l1=%g" % l for l in lams)))
for a in arms:
    row = []
    for lam in lams:
        v = [len(mem[(s, a, lam)]) for s in seeds if (s, a, lam) in mem]
        row.append("%7.0f" % (st.mean(v) if v else 0))
    print("%-12s %s" % (a, "  ".join(row)))
