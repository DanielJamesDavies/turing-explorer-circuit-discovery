"""Readable qualitative exhibit for all six Gemma seeds.

For each seed, three views that together show the SELECTION difference
(the confirmed finding), rather than sorting by alpha (which does NOT
split by echo/context -- see echo_vs_context.md):

  1. their top-5 direct-edge features, and whether each is in our set
  2. OUR members that their ranking buries (rank > 300) -- the
     population that distinguishes the two methods
  3. our members they rank highly (their top-30) -- the shared core

  python qualitative_examples.py -> qualitative_examples.md (+ stdout)
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
BURY = 300
TOP = 30

LABELS = {}
for line in open(HERE / "neuronpedia_labels.jsonl"):
    r = json.loads(line)
    LABELS[(r["layer"], r["feat"])] = r["label"]

RANK, ORDER = {}, {}
for line in open(HERE / "theirs_gtc_nodes.jsonl"):
    r = json.loads(line)
    key = (int(r["layer"]), int(r["latent"]))
    RANK[key] = {(int(l), int(f)): i + 1
                 for i, (l, f, *_) in enumerate(r["ranking"])}
    ORDER[key] = [(int(l), int(f)) for l, f, *_ in r["ranking"]]

STOP = set("""the and for with that this from into over under about above
a an of in on to as at by or nor but if then than so such when while
where which who whom whose what how why all any both each few more most
other some only own same too very can will just don should now is are
was were be been being have has had do does did doing would could may
might must shall its it their there here they them these those you your
word words phrase phrases term terms text texts mention mentions
mentioned related relating reference references referring instance
instances including include includes especially particularly various
seemingly unrelated specific context contexts use used using usage
associated indicating indicate indicates one two three sometimes often
also written form forms name names number numbers""".split())
TOKQ = re.compile(r'"([^"]+)"')
cw = lambda s: {w for w in re.findall(r"[a-z]{4,}", s.lower()) if w not in STOP}
qt = lambda s: {q.strip().lower() for q in TOKQ.findall(s)}


def main():
    out = ["# Qualitative examples: six Gemma seeds", "",
           "`E` = echo (label shares vocabulary with the seed's label), "
           "`C` = context. Ranks are positions in circuit-tracer's "
           "direct-edge ranking over ~20k features. Labels are "
           "Neuronpedia auto-interp and inherit its errors.", ""]
    for line in open(HERE / "ours_gtc_members.jsonl"):
        r = json.loads(line)
        if r["arm"] != "triamp400":
            continue
        key = (int(r["layer"]), int(r["latent"]))
        slab = LABELS.get(key, "(no label)")
        sc, sq = cw(slab), qt(slab)
        cls = lambda k: ("E" if (cw(LABELS.get(k, "")) & sc)
                         or (qt(LABELS.get(k, "")) & sq) else "C")
        alphas = {(int(l), int(f)): a
                  for l, d in r["alphas"].items() for f, a in d.items()}
        mine = set(alphas)
        out += ["## L%d/%d — *%s*" % (key[0], key[1], slab), "",
                "our circuit: %d nodes" % len(mine), "",
                "**What attribution ranks highest** (their top 5):", "",
                "| their rank | feature | E/C | in our set | label |",
                "|---|---|---|---|---|"]
        for i, k in enumerate(ORDER[key][:5], 1):
            out.append("| %d | L%d/%d | %s | %s | %s |" % (
                i, k[0], k[1], cls(k), "yes" if k in mine else "—",
                LABELS.get(k, "(unlabelled)")))
        buried = sorted((RANK[key].get(k, 10 ** 9), k) for k in mine
                        if RANK[key].get(k, 10 ** 9) > BURY)
        out += ["", "**What we include that they bury** (their rank > %d): "
                "%d of %d nodes" % (BURY, len(buried), len(mine)), ""]
        if buried:
            out += ["| their rank | feature | E/C | alpha | label |",
                    "|---|---|---|---|---|"]
            for rk, k in buried[:8]:
                out.append("| %s | L%d/%d | %s | %.2f | %s |" % (
                    "not in 20k" if rk == 10 ** 9 else rk, k[0], k[1],
                    cls(k), alphas[k], LABELS.get(k, "(unlabelled)")))
            if len(buried) > 8:
                out.append("| … | (%d more) | | | |" % (len(buried) - 8))
        else:
            out.append("*(none — the whole circuit sits in their head)*")
        shared = sorted((RANK[key][k], k) for k in mine
                        if RANK[key].get(k, 10 ** 9) <= TOP)
        out += ["", "**Shared core** (our members in their top %d): %d nodes"
                % (TOP, len(shared)), ""]
        if shared:
            out += ["| their rank | feature | E/C | alpha | label |",
                    "|---|---|---|---|---|"]
            for rk, k in shared[:6]:
                out.append("| %d | L%d/%d | %s | %.2f | %s |" % (
                    rk, k[0], k[1], cls(k), alphas[k],
                    LABELS.get(k, "(unlabelled)")))
        out.append("")
    (HERE / "qualitative_examples.md").write_text("\n".join(out),
                                                  encoding="utf-8", newline="")
    print("\n".join(out))


if __name__ == "__main__":
    main()
