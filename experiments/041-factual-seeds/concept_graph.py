"""Read a VALIDATED circuit as a causal graph over concepts.

The premise, stated so it can be attacked: a transcoder latent is
treated as standing for a concept, named by its Neuronpedia auto-interp
label. Our circuit for a seed latent is the set of upstream latents
whose amplification reconstructs it, and which the null suite says is
not reachable by chance. So -- IF the latent/concept identification
holds -- the circuit is a causal claim: "these concepts compose into
that concept."

What this does NOT establish: that the label is correct (auto-interp
errs), that a latent is one concept rather than several, or that the
composition is the model's only route to the seed. Circuits are
sufficient sets, not unique ones ([[l2-crossover-universal-core]]).
Those caveats travel with every table this writes.

Only seeds whose circuit PASSES the two-criteria bar are rendered:
zero-fill and mean-fill faithfulness in [0.8, 1.25] and necessity above
the run's nulls. A circuit that does not reconstruct its seed licenses
no claim about concepts at all.

  python concept_graph.py            -> concept_graphs.md
"""
import json
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
GEMMA = HERE.parent / "038-transcoder-compare-gemma"
BAND = (0.8, 1.25)
TAG = os.environ.get("ROWS_TAG", "_fact")

LABELS = {}
for fn in (HERE / "labels.jsonl", GEMMA / "neuronpedia_labels.jsonl"):
    if fn.exists():
        for line in open(fn):
            r = json.loads(line)
            LABELS[(r["layer"], r["feat"])] = r["label"]

rows = defaultdict(dict)
rp = GEMMA / ("ours_gtc%s_rows.jsonl" % TAG)
for line in open(rp):
    r = json.loads(line)
    rows[(r["layer"], r["latent"])][r["arm"]] = r

mem = {}
mp = GEMMA / ("ours_gtc%s_members.jsonl" % TAG)
for line in open(mp):
    r = json.loads(line)
    if r["arm"] == "triamp400":
        mem[(int(r["layer"]), int(r["latent"]))] = {
            (int(l), int(f)): float(a)
            for l, d in r["alphas"].items() for f, a in d.items()}

out = ["# Concept graphs from validated circuits", "",
       "Each seed latent is named by its Neuronpedia auto-interp label and",
       "its circuit is listed as the upstream latents that reconstruct it.",
       "GATE: only circuits with zero-fill AND mean-fill faithfulness in",
       "[%.2f, %.2f] and necessity above the run's nulls appear as validated;"
       % BAND, "others are listed with their numbers and marked NOT VALIDATED.",
       "",
       "Caveats that travel with every table: auto-interp labels can be",
       "wrong or vague; a latent need not be exactly one concept; and a",
       "circuit is a SUFFICIENT set, not the model's only route to the seed.",
       ""]

nval = 0
for key in sorted(mem, key=lambda k: (k[0], k[1])):
    r = rows[key].get("triamp400")
    if not r:
        continue
    nulls = [v.get("ampF0") for a, v in rows[key].items()
             if a.startswith("null") and v.get("ampF0") is not None]
    f0, fm, sup = r.get("ampF0"), r.get("ampFM"), r.get("sup")
    ok = (f0 is not None and fm is not None
          and BAND[0] <= f0 <= BAND[1] and BAND[0] <= fm <= BAND[1]
          and (not nulls or f0 > max(nulls)))
    nval += bool(ok)
    slab = LABELS.get(key, "(no label)")
    out += ["## L%d/%d — *%s*" % (key[0], key[1], slab), "",
            "%s | n=%d | zero-fill %.2f | mean-fill %.2f | necessity %.2f"
            " | best null %s"
            % ("**VALIDATED**" if ok else "NOT VALIDATED — no concept claim",
               r["n"], f0 if f0 is not None else float("nan"),
               fm if fm is not None else float("nan"),
               sup if sup is not None else float("nan"),
               ("%.2f" % max(nulls)) if nulls else "n/a"), ""]
    if not ok:
        out.append("")
        continue
    out += ["| lyr | feat | alpha | concept (auto-interp label) |",
            "|---|---|---|---|"]
    for (l, f), a in sorted(mem[key].items(), key=lambda kv: -kv[1]):
        out.append("| %d | %d | %.2f | %s |"
                   % (l, f, a, LABELS.get((l, f), "(unlabelled)")))
    out.append("")

out += ["---", "", "%d of %d circuits validated." % (nval, len(mem))]
(HERE / "concept_graphs.md").write_text("\n".join(out), encoding="utf-8",
                                        newline="")
print("\n".join(out[:40]))
print("\nwrote concept_graphs.md (%d validated of %d)" % (nval, len(mem)))
