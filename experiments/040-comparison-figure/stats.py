"""Statistics behind the comparison figure, from points.jsonl.

Per arena:
  1. MATCHED-SIZE pairing: for each external method, per-seed win/loss
     vs tri-amp on zero-fill faithfulness at the size closest to ours,
     with a sign-test style count (6 seeds is too few for p-values worth
     quoting; the count and the median difference are reported).
  2. NODES-TO-BAND: per method per seed, the smallest measured size
     whose zero-fill AND mean-fill both land in [0.8, 1.25]; the
     distribution and the median ratio to tri-amp's size. "never" =
     no measured size reached the band (right-censored at the largest
     measured size, stated as > that size).
  3. Bootstrap 95% CI (10k resamples over seeds) for each method's
     matched-size median f0.

Rerun after collect.py whenever new arms land.

  python stats.py    ->  stats.md (+ stdout)
"""
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
BAND = (0.8, 1.25)


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def boot_ci(vals, it=10000, seed=7):
    if len(vals) < 2:
        return (None, None)
    rng = random.Random(seed)
    meds = sorted(med([rng.choice(vals) for _ in vals]) for _ in range(it))
    return meds[int(0.025 * it)], meds[int(0.975 * it)]


def main():
    pts = [json.loads(l) for l in open(HERE / "points.jsonl")]
    out = ["# Comparison statistics", ""]
    for arena in ("gemma-tc", "llama-tc", "turingllm"):
        ap = [p for p in pts if p["arena"] == arena]
        by = defaultdict(lambda: defaultdict(list))   # method -> seed -> pts
        for p in ap:
            by[p["method"]][p["seed"]].append(p)
        ours = by.get("tri-amp", {})
        our_size = {s: med([p["n"] for p in v]) for s, v in ours.items()}
        our_best = {s: max((p["f0"] for p in v if p["f0"] is not None),
                           default=None) for s, v in ours.items()}
        # ours' own nodes-to-band and matched f0 (point nearest own median size)
        def nearest(v, n0):
            return min(v, key=lambda p: abs(p["n"] - n0))
        our_matched = {s: nearest(v, our_size[s])["f0"] for s, v in ours.items()}

        def ntb(v):
            ok = [p["n"] for p in v
                  if p["f0"] is not None and p["fm"] is not None
                  and BAND[0] <= p["f0"] <= BAND[1] and BAND[0] <= p["fm"] <= BAND[1]]
            return min(ok) if ok else None

        out += ["## %s" % arena, "",
                "| method | matched-size f0 med [95%CI] | wins-losses vs tri-amp (|f0-1|) | nodes-to-band med | ntb / tri-amp size |",
                "|---|---|---|---|---|"]
        rows = []
        for m, seeds in sorted(by.items()):
            matched, wins, losses, ntbs, ratios = [], 0, 0, [], []
            for s, v in seeds.items():
                if s not in our_size:
                    continue
                pm = nearest(v, our_size[s])
                # a comparison "at matched size" is only meaningful if the
                # method actually has a point near ours (within 3x); home
                # arms whose smallest cut is 400x away are excluded here
                # and speak through nodes-to-band instead.
                if pm["f0"] is not None and our_size[s]                         and 1 / 3 <= pm["n"] / our_size[s] <= 3:
                    matched.append(pm["f0"])
                    if m != "tri-amp":
                        # closer to unity wins: overshoot is failure too
                        if abs(pm["f0"] - 1) < abs(our_matched[s] - 1):
                            wins += 1
                        elif abs(pm["f0"] - 1) > abs(our_matched[s] - 1):
                            losses += 1
                b = ntb(v)
                ntbs.append(b)
                if b is not None and our_size[s]:
                    ratios.append(b / our_size[s])
            lo, hi = boot_ci(matched)
            never = sum(1 for b in ntbs if b is None)
            nb = med([b for b in ntbs if b is not None])
            rows.append((m, matched, lo, hi, wins, losses, len(seeds), nb, never, ratios))
        for (m, matched, lo, hi, w, l, ns, nb, never, ratios) in rows:
            out.append("| %s | %s [%s, %s] | %s | %s%s | %s |" % (
                m, "%.2f" % med(matched) if matched else "-",
                "%.2f" % lo if lo is not None else "-",
                "%.2f" % hi if hi is not None else "-",
                ("%d-%d of %d" % (w, l, ns)) if m != "tri-amp" else "(ref)",
                ("%d" % nb) if nb is not None else "never",
                (" (%d/%d never)" % (never, ns)) if never else "",
                ("%.1fx" % med(ratios)) if ratios else "-"))
        out.append("")
    text = "\n".join(out)
    (HERE / "stats.md").write_text(text, encoding="utf-8", newline="")
    print(text)


if __name__ == "__main__":
    main()
