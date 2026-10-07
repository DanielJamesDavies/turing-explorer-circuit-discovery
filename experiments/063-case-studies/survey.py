"""CATALOGUE of passing late-layer circuits for the case-study hunt: one compact entry per target, split into subsets.

Unlike inspect_circuit.py's deepconcept screen, nothing is filtered on peak-token consistency: calculation and
knowledge latents are often token-consistent (digits, operators, names), and they are what this hunt looks for.

Per target (context cache and decoder only, no circuit pass):
  scores (Z / A / C, necessity, induce, size, amplifier flag), peak tokens and consistency, distinct windows from its
  strongest contexts with the peak marked, a contrast snippet, its logit effect, and theme tags from simple keyword
  counts over its windows and logit tokens (MATH, LOGIC, KNOWLEDGE, CODE, SCIENCE, ...). Tags are a reading aid for
  the agents, not a classification: the windows are the evidence.

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/063-case-studies/survey.py
  env: MIN_LAYER (6)  N_SUBSETS (10)  N_WIN (4)  TAG_MIN (0.5 hits per window)
       RETAG=1 recomputes the tags on the existing catalogue and rewrites the subsets (no model pass)
  ->   results_lab/catalogue.jsonl, results_lab/subsets/subset_NN.md
"""
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("OUT", str(HERE.parent / "062-h100-protocol-v1" / "out_full" / "out"))
import inspect_circuit as I  # noqa: E402  (reads OUT at import)

LAB = HERE / "results_lab"
MIN_LAYER = int(os.environ.get("MIN_LAYER", 6))
N_SUBSETS = int(os.environ.get("N_SUBSETS", 10))
N_WIN = int(os.environ.get("N_WIN", 4))

# keyword themes, matched case-insensitively on whole words in the windows and logit tokens (a reading aid only)
THEMES = {
    "MATH": r"equation|sum|product|multipl\w*|divid\w*|fraction|integer|derivative|integral|solve|solving|calculat\w*|"
            r"percent\w*|formula|variable|quadratic|polynomial|matrix|vector|theorem|proof|prime|factor\w*|"
            r"remainder|digit|decimal|ratio|probability|algebra\w*|geometr\w*|angle|triangle|area|volume|mean|median",
    "LOGIC": r"if|then|therefore|thus|hence|implies|because|since|not|all|some|none|every|either|neither|unless|"
             r"premise|conclusion|contradiction|valid|true|false|syllogism|deduc\w*|infer\w*|consequently",
    "KNOWLEDGE": r"capital|century|invented|discovered|founded|located|known|born|died|empire|king|queen|war|"
                 r"president|country|city|river|author|wrote|named|dynasty|revolution|treaty|ancient",
    "SCIENCE": r"atom\w*|molecul\w*|electron\w*|cell\w*|protein\w*|gene\w*|energy|force|mass|acid\w*|reaction\w*|"
               r"species|evolution|photosynthesis|gravity|velocity|chemical|element\w*|enzyme\w*|dna",
    "CODE": r"def|return|function|import|class|variable|loop|array|string|int|print|python|java|code|syntax",
}
THEME_RE = {k: re.compile(r"\b(?:%s)\b" % v, re.I) for k, v in THEMES.items()}
# hyphens and slashes are left out of MATH: prose is full of them
SYMBOL = {"MATH": re.compile(r"[0-9=+*^×÷<>√π∑]"), "CODE": re.compile(r"[{}();\[\]_]")}


def tags(windows, logit_tokens):
    """Theme -> hits PER WINDOW, counted over the windows and the logit tokens; tags with >= TAG_MIN hits/window."""
    joined = " \n ".join(windows + logit_tokens).replace("[[", "").replace("]]", "")   # the peak marker is not code
    n = max(1, len(windows))
    out = {}
    for k, rx in THEME_RE.items():
        hits = len(rx.findall(joined))
        if k in SYMBOL:
            hits += len(SYMBOL[k].findall(joined)) / 3.0            # symbols are dense; weight them down
        out[k] = hits / n
    return {k: round(v, 2) for k, v in sorted(out.items(), key=lambda kv: -kv[1]) if v >= TAG_MIN}


TAG_MIN = float(os.environ.get("TAG_MIN", 0.5))


def write_subsets(rows):
    """Subsets: round-robin over the (layer desc, kind) order, so every subset spans layers 6-11 and all kinds."""
    (LAB / "subsets").mkdir(parents=True, exist_ok=True)
    for s in range(N_SUBSETS):
        part = rows[s::N_SUBSETS]
        lines = ["# Subset %02d: %d passing circuits (layer >= %d)" % (s, len(part), MIN_LAYER), "",
                 "Format: key | size | Z A C | necessity | induce | amp | consistency | peaks | tags / "
                 "then windows ([[peak]]), logit effect.", ""]
        for r in part:
            lines.append("## %s | n %d | Z %.2f A %.2f C %.2f | nec %.2f | ind %.2f | %s | cons %.0f%% | %s | %s" % (
                r["key"], r["n"], r["Z"], r["A"], r["C"], r["nec"], r["induce"], "AMP" if r["amp"] else "-",
                100 * r["consistency"], ", ".join("%r×%d" % tuple(p) for p in r["peaks"][:3]),
                " ".join("%s:%.1f" % kv for kv in r["tags"].items()) or "no tags"))
            lines += ["- …%s" % w for w in r["windows"]]
            lines.append("- logits + %s | − %s" % (r["logit_up"], r["logit_down"]))
        (LAB / "subsets" / ("subset_%02d.md" % s)).write_text("\n".join(lines) + "\n", encoding="utf-8")


def retag():
    """RETAG=1: recompute tags on the existing catalogue (no model pass) and rewrite it and the subsets."""
    rows = [json.loads(l) for l in open(LAB / "catalogue.jsonl", encoding="utf-8")]
    for r in rows:
        r["tags"] = tags(r["windows"], r["logit_up"])
    with open(LAB / "catalogue.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_subsets(rows)
    print("retagged %d entries; tag counts %s" % (len(rows), dict(Counter(t for r in rows for t in r["tags"]))))


def main():
    d = I.scores_table()
    keys = d[d.passes & ~d.near & (d.layer >= MIN_LAYER)].sort_values(["layer", "kind"], ascending=[False, True]).index
    print("passing, non-near-threshold, layer >= %d: %d targets" % (MIN_LAYER, len(keys)), flush=True)
    B = I.Browser()
    have = {p.stem for p in (HERE / "results_full_deep" / "reports").glob("*.md")}
    LAB.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, key in enumerate(keys):
        tl, tk, ti = I.parse(key)
        rec = torch.load(I.OUT / "ctx" / ("%s.pt" % key), weights_only=False)[key]
        pos, arg, neg = rec["strong"]["pos"], rec["strong"]["arg"].long(), rec["neg"]
        peaks = Counter(B.dec([pos[b, int(arg[b])]]) for b in range(pos.shape[0]))
        top = peaks.most_common(5)
        wins = B.distinct((B.window(pos[b].tolist(), int(arg[b]), before=14, after=4) for b in range(pos.shape[0])),
                          N_WIN)
        up, down = B.logits(tl, tk, ti)
        r = d.loc[key]
        rows.append(dict(
            key=key, layer=int(tl), kind=tk, n=int(r.n), amp=bool(r.amp_any) if r.amp_any == r.amp_any else None,
            Z=round(float(r.free0_tk), 2), A=round(float(r.freeM_topk_tk), 2), C=round(float(r.freeN_topk_tk), 2),
            nec=round(float(r.phi_sup_blind_tk), 2), induce=round(float(r.phi_cf_alpha_blind_tk), 2),
            peaks=[[t, c] for t, c in top], consistency=round(top[0][1] / pos.shape[0], 2), windows=wins,
            contrast=B.dec(neg[0, 24:44].tolist()), logit_up=up, logit_down=down,
            tags=tags(wins, up), report=key in have))
        if (n + 1) % 200 == 0:
            print("  %d/%d" % (n + 1, len(keys)), flush=True)
    with open(LAB / "catalogue.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_subsets(rows)
    tagc = Counter(t for r in rows for t in r["tags"])
    print("wrote %d entries, %d subsets; tag counts %s; reports already present %d"
          % (len(rows), N_SUBSETS, dict(tagc), sum(r["report"] for r in rows)), flush=True)


if __name__ == "__main__":
    retag() if os.environ.get("RETAG") else main()
