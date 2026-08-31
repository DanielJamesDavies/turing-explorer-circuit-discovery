"""THE ECHO TEST -- the quantitative form of the qualitative finding.

Claim under test: attribution ranks highest those circuit members that
merely RESTATE the seed concept (its token, its synonyms -- "echoes"),
while our amplitude fit puts its weight on CONTEXT members that supply
the seed's preconditions and share no vocabulary with it.

Method: label-lexical. A member is an ECHO if its Neuronpedia label
shares a content word (>=4 chars, stopwords and generic interp-jargon
removed) with the SEED's own label, or if either label quotes the same
token. Everything else is CONTEXT. This is deliberately crude and
label-derived: it is a descriptive statistic over auto-interp text, not
a semantic ground truth, and it inherits any auto-interp error. It is
reported as such.

Then, per seed and pooled: median alpha and median attribution rank for
each class, plus a rank-sum test (Mann-Whitney U, normal approximation)
on alpha between classes.

  python echo_vs_context.py -> echo_vs_context.md + echo_vs_context.pdf/png
"""
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
STOP = set("""the and for with that this from into over under about above
们 a an of in on to as at by or nor but if then than so such when while
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


def content(label):
    ws = {w for w in re.findall(r"[a-z]{4,}", label.lower()) if w not in STOP}
    return ws


def quoted(label):
    return {q.strip().lower() for q in TOKQ.findall(label)}


def mwu(a, b):
    """Mann-Whitney U on (a > b), normal approximation with tie
    correction; returns (U, z, p_two_sided). Small-n caveat applies."""
    if not a or not b:
        return None, None, None
    allv = sorted(a + b)
    def rank(x):
        i = allv.index(x)
        j = len(allv) - 1 - allv[::-1].index(x)
        return (i + j) / 2 + 1
    ra = sum(rank(x) for x in a)
    na, nb = len(a), len(b)
    U = ra - na * (na + 1) / 2
    mu = na * nb / 2
    sd = math.sqrt(na * nb * (na + nb + 1) / 12)
    if sd == 0:
        return U, None, None
    z = (U - mu) / sd
    p = math.erfc(abs(z) / math.sqrt(2))
    return U, z, p


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


LABELS = {}
for line in open(HERE / "neuronpedia_labels.jsonl"):
    r = json.loads(line)
    LABELS[(r["layer"], r["feat"])] = r["label"]

RANK = {}
for line in open(HERE / "theirs_gtc_nodes.jsonl"):
    r = json.loads(line)
    RANK[(int(r["layer"]), int(r["latent"]))] = {
        (int(l), int(f)): i + 1 for i, (l, f, *_) in enumerate(r["ranking"])}

out = ["# The echo test: what attribution ranks vs what the fit amplifies",
       "",
       "ECHO = member label shares a content word or a quoted token with",
       "the SEED's label. CONTEXT = everything else. Label-lexical and",
       "descriptive: it is a statistic over auto-interp text, not semantic",
       "ground truth.", "",
       "| seed | seed label | n | echo | context | med alpha echo | med alpha ctx |"
       " med ct-rank echo | med ct-rank ctx | MWU p (alpha) |",
       "|---|---|---|---|---|---|---|---|---|---|"]
pool = {"echo_a": [], "ctx_a": [], "echo_r": [], "ctx_r": []}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

for line in open(HERE / "ours_gtc_members.jsonl"):
    r = json.loads(line)
    if r["arm"] != "triamp400":
        continue
    key = (int(r["layer"]), int(r["latent"]))
    slab = LABELS.get(key, "")
    sc, sq = content(slab), quoted(slab)
    ea, ca, er, cr = [], [], [], []
    for l, d in r["alphas"].items():
        for f, a in d.items():
            k = (int(l), int(f))
            lab = LABELS.get(k, "")
            is_echo = bool(content(lab) & sc) or bool(quoted(lab) & sq)
            rk = RANK[key].get(k)
            (ea if is_echo else ca).append(float(a))
            if rk:
                (er if is_echo else cr).append(rk)
    _, _, p = mwu(ca, ea)
    out.append("| L%d/%d | %s | %d | %d | %d | %s | %s | %s | %s | %s |" % (
        key[0], key[1], slab[:44], len(ea) + len(ca), len(ea), len(ca),
        "%.2f" % med(ea) if ea else "-", "%.2f" % med(ca) if ca else "-",
        med(er) or "-", med(cr) or "-",
        ("%.3f" % p) if p is not None else "-"))
    pool["echo_a"] += ea; pool["ctx_a"] += ca
    pool["echo_r"] += er; pool["ctx_r"] += cr

# CONTROL: "our set is 80% context" means nothing without the base rate.
# Compare the echo share of OUR n members against THEIR top-n (matched
# size, same seed, same label source) and against a random draw from
# their full 20k ranking.
share_ours, share_head = [], []
out += ["", "## Control: echo share, ours vs their matched head", "",
        "| seed | ours echo share | their top-n echo share | their 20k echo share |",
        "|---|---|---|---|"]
import random
rng = random.Random(11)
for line in open(HERE / "ours_gtc_members.jsonl"):
    r = json.loads(line)
    if r["arm"] != "triamp400":
        continue
    key = (int(r["layer"]), int(r["latent"]))
    slab = LABELS.get(key, "")
    sc, sq = content(slab), quoted(slab)
    def is_echo(k):
        lab = LABELS.get(k)
        if lab is None:
            return None                      # unlabelled: excluded, stated
        return bool(content(lab) & sc) or bool(quoted(lab) & sq)
    mem = [(int(l), int(f)) for l, d in r["members"].items() for f in d]
    n = len(mem)
    ours_e = [is_echo(k) for k in mem]
    ours_share = sum(1 for x in ours_e if x) / max(1, len(ours_e))
    order = sorted(RANK[key], key=lambda k: RANK[key][k])
    head = [is_echo(k) for k in order[:n]]
    head = [x for x in head if x is not None]
    rnd = [is_echo(k) for k in rng.sample(order, min(len(order), 400))]
    rnd = [x for x in rnd if x is not None]
    if head:
        share_ours.append(ours_share); share_head.append(sum(head) / len(head))
    out.append("| L%d/%d | %.0f%% (n=%d) | %s | %s |" % (
        key[0], key[1], 100 * ours_share, n,
        ("%.0f%% (labelled %d/%d)" % (100 * sum(head) / len(head), len(head), n))
        if head else "no labels",
        ("%.0f%% (labelled %d)" % (100 * sum(rnd) / len(rnd), len(rnd)))
        if rnd else "no labels"))
nlo = sum(1 for a, b in zip(share_ours, share_head) if a < b)
out += ["",
        "Their matched head is fully labelled (fetch_labels.py fetches it),",
        "so the middle column is a like-for-like base rate, not a subset.",
        "The 20k column samples 400 features and counts only those already",
        "cached, so it is indicative only.",
        "",
        "OUR echo share is lower than their matched head on %d of %d seeds"
        % (nlo, len(share_ours)),
        "(sign test p=%.3f); ratio of shares, median %.2f -- our circuits"
        % (2 ** -len(share_ours) * 2 * sum(
            math.comb(len(share_ours), k)
            for k in range(nlo, len(share_ours) + 1)),
           med([a / b for a, b in zip(share_ours, share_head) if b > 0])),
        "carry roughly half the seed-vocabulary density of the equally",
        "sized attribution head.", ""]

_, z, p = mwu(pool["ctx_a"], pool["echo_a"])
_, zr, pr = mwu(pool["echo_r"], pool["ctx_r"])
out += ["",
        "## Pooled over six seeds", "",
        "| class | n | median alpha | median ct-rank |", "|---|---|---|---|",
        "| echo | %d | %.2f | %d |" % (len(pool["echo_a"]),
                                       med(pool["echo_a"]), med(pool["echo_r"])),
        "| context | %d | %.2f | %d |" % (len(pool["ctx_a"]),
                                          med(pool["ctx_a"]), med(pool["ctx_r"])),
        "",
        "Mann-Whitney on alpha (context > echo): z=%.2f, p=%.4g" % (z, p),
        "Mann-Whitney on attribution rank (echo better i.e. lower): "
        "z=%.2f, p=%.4g" % (zr, pr),
        "",
        "Caveat: members are not independent (six circuits, shared",
        "features across seeds), so p-values are indicative, not a",
        "licence to claim significance at a seed level.",
        ]

ax[0].boxplot([pool["echo_a"], pool["ctx_a"]], tick_labels=["echo", "context"],
              showfliers=False, patch_artist=True,
              boxprops=dict(facecolor="#1657d6", alpha=0.35))
ax[0].set_ylabel("our fitted alpha")
ax[0].set_title("What the fit amplifies", fontsize=10)
ax[1].boxplot([pool["echo_r"], pool["ctx_r"]], tick_labels=["echo", "context"],
              showfliers=False, patch_artist=True,
              boxprops=dict(facecolor="#c22f2f", alpha=0.35))
ax[1].set_yscale("log")
ax[1].set_ylabel("rank in their direct-edge ranking (log)")
ax[1].set_title("What attribution ranks highly", fontsize=10)
for a in ax:
    a.grid(alpha=0.15, axis="y")
fig.suptitle("Echo vs context members of six Gemma tri-amp circuits: the two "
             "methods weight opposite populations", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
for ext in ("pdf", "png"):
    fig.savefig(HERE / ("echo_vs_context.%s" % ext), dpi=170)

(HERE / "echo_vs_context.md").write_text("\n".join(out), encoding="utf-8",
                                         newline="")
print("\n".join(out))
