"""Summarise out/rows.jsonl + out/calibration.jsonl from check.py; decode a few examples. Writes out/report.md.

  .venv/bin/python -I experiments/113-paraphrase-check/report.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "out"


def load(p):
    return [json.loads(l) for l in open(p)]


def q(x):
    x = np.asarray(x)
    return "%.3f / %.3f / %.3f / %.3f" % tuple(np.percentile(x, [10, 50, 90, 99]))


def main():
    rows, cal = load(OUT / "rows.jsonl"), load(OUT / "calibration.jsonl")
    L = []
    cp, cu = np.array([c["cos_para"] for c in cal]), np.array([c["cos_unrel"] for c in cal])
    L.append("## Calibration (%d originals: one window vs its paraphrase file / vs an unrelated paraphrase file)\n" % len(cal))
    L.append("| | p10 / p50 / p90 / p99 |\n|---|---|")
    L.append("| max cos, own paraphrase file | %s |" % q(cp))
    L.append("| max cos, unrelated file | %s |" % q(cu))
    L.append("| 8-gram share of best paraphrase match | %s |" % q([c["ng8_para"] for c in cal]))
    tau = float(np.percentile(cu, 99))
    L.append("\ntau = p99 of unrelated = %.3f; recall of known paraphrases at tau = %.1f%%\n" % (tau, 100 * (cp >= tau).mean()))

    c = np.array([r["cos"] for r in rows]); cx = np.array([r["cos_other_target"] for r in rows])
    cr = np.array([r["cos_random"] for r in rows]); g = np.array([r["ng8_max"] for r in rows])
    keys = sorted({r["key"] for r in rows})
    L.append("## Held-out strongest rows (%d rows, %d targets): max over the target's 96 training-side rows\n" % (len(rows), len(keys)))
    L.append("| comparison | TF-IDF cos p10 / p50 / p90 / p99 | share >= tau |\n|---|---|---|")
    L.append("| same target's training rows | %s | %.1f%% |" % (q(c), 100 * (c >= tau).mean()))
    L.append("| another target's training rows | %s | %.1f%% |" % (q(cx), 100 * (cx >= tau).mean()))
    L.append("| random corpus windows | %s | %.1f%% |" % (q(cr), 100 * (cr >= tau).mean()))
    L.append("\n8-gram share (max over training rows): p50 %.3f, p99 %.3f, max %.3f; >= 0.5: %d rows\n"
             % (np.median(g), np.percentile(g, 99), g.max(), int((g >= 0.5).sum())))

    def share(k, mask=None):
        v = np.array([r[k] for r in rows])
        return 100 * v.mean()
    L.append("## Provenance of held-out rows vs the target's training side\n")
    L.append("| held row's ... also contributes a training row | share of held rows |\n|---|---|")
    L.append("| source file | %.1f%% |" % share("any_same_file"))
    L.append("| document group (original + its paraphrase file) | %.1f%% |" % share("any_same_doc"))
    L.append("| topic group (the 6 sibling documents of one leaf topic) | %.1f%% |" % share("any_same_topic"))
    L.append("| held row is from a paraphrase file | %.1f%% |" % share("held_aug"))
    L.append("\nBest TF-IDF match (argmax) is in the same file %.1f%%, same doc group %.1f%%, same topic %.1f%%.\n"
             % (share("match_same_file"), share("match_same_doc"), share("match_same_topic")))
    sd = np.array([r["match_same_doc"] for r in rows])
    L.append("cos of the best match: same doc group median %.3f (n %d), other median %.3f (n %d)\n"
             % (np.median(c[sd]), sd.sum(), np.median(c[~sd]), (~sd).sum()))
    para = c >= tau
    L.append("**Paraphrase-like (cos >= tau): %d of %d rows (%.1f%%); targets with >= 1: %d of %d; of these rows, best match in same doc group %.1f%%.**\n"
             % (para.sum(), len(rows), 100 * para.mean(), len({r["key"] for r, p in zip(rows, para) if p}), len(keys),
                100 * sd[para].mean() if para.any() else 0))
    # per target: held rows left if doc-disjoint / topic-disjoint
    per = {}
    for r in rows:
        d = per.setdefault(r["key"], [0, 0, 0, 0])
        d[0] += 1; d[1] += not r["any_same_doc"]; d[2] += not r["any_same_topic"]; d[3] += not r["any_same_file"]
    a = np.array(list(per.values()))
    L.append("## Held-out rows that survive a group rule (no change to training rows)\n")
    L.append("| rule | median left of 16 | targets < 8 |\n|---|---|---|")
    for n, j in (("file-disjoint", 3), ("doc-group-disjoint", 1), ("topic-disjoint", 2)):
        L.append("| %s | %d | %.1f%% |" % (n, np.median(a[:, j]), 100 * (a[:, j] < 8).mean()))

    # examples: highest-cos held rows, distinct targets
    from transformers import AutoTokenizer
    T = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", use_fast=True, local_files_only=True)
    import sys
    sys.path.insert(0, str(HERE))
    from check import window, groups
    paths = groups()[0]
    L.append("\n## Examples (held-out row / best training match)\n")
    seen, ex = set(), []
    for i in np.argsort(-c):
        r = rows[i]
        if r["key"] in seen or r["ng8_max"] >= 0.5:
            continue
        seen.add(r["key"]); ex.append(r)
        if len(ex) == 3:
            break
    mid = [rows[i] for i in np.argsort(-c) if abs(c[i] - tau) < 0.02][:2]
    for r in ex + mid:
        L.append("**%s** cos %.3f, 8-gram %.2f; held src %d `%s`; match src %d `%s`\n"
                 % (r["key"], r["cos"], r["ng8"], r["held_src"], paths[r["held_src"]], r["match_src"], paths[r["match_src"]]))
        L.append("- held: %s" % json.dumps(T.decode(window(r["held_id"]).tolist())))
        L.append("- train: %s\n" % json.dumps(T.decode(window(r["match_id"]).tolist())))
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
