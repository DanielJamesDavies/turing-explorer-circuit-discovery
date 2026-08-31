"""CONSTRUCTED KNOWLEDGE BEHAVIOURS (SFC-style clusters, but each
cluster = one FACT expressed as many paraphrases):

  kb0  special relativity -> 1905
  kb1  general relativity -> 1915   (minimal pair with kb0)
  kb2  relativity proposed by -> Einstein (attribution retrieval)

For each paraphrase and each split point j of the continuation's
tokens, candidate task = (prompt + continuation[:j] -> token j). The
model verifies each candidate (left-pad to the 63-token window frame
used by behaviour_runner; p(target) measured IN that frame, so the
padding is validated, not assumed). Per behaviour we keep the split
point with the most passing paraphrases. Writes
knowledge_clusters.pt in the behaviour_clusters.pt schema. Exits
nonzero if any behaviour keeps < MINKEEP contexts.

  PYTHONPATH=src python .../make_knowledge.py
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from hardware import detect_devices, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer

HERE = Path(__file__).parent
SEQ = 64
ANCHOR = SEQ - 2
PKEEP = float(os.environ.get("PKEEP", 0.10))
MINKEEP = int(os.environ.get("MINKEEP", 12))

SR = [
    "Einstein published his theory of special relativity in the year",
    "Einstein's theory of special relativity was published in",
    "The special theory of relativity was first published in",
    "Special relativity was introduced by Einstein in",
    "Einstein first presented special relativity in",
    "The theory of special relativity dates from",
    "Einstein's paper on special relativity appeared in",
    "The special theory of relativity appeared in the year",
    "Einstein proposed special relativity in",
    "Special relativity was proposed in the year",
    "The publication of special relativity came in",
    "Einstein's special theory of relativity emerged in",
    "The famous paper on special relativity was published in",
    "Physics changed forever when special relativity appeared in",
    "Einstein wrote his special relativity paper in",
    "The special relativity paper was released in",
    "Einstein developed special relativity in the year",
    "His theory of special relativity was announced in",
    "The theory we call special relativity arrived in",
    "Einstein completed his special theory of relativity in",
    "Special relativity, published in",
    "The year Einstein published special relativity was",
    "Einstein unveiled special relativity in",
    "Modern physics began with special relativity in",
]
GR = [
    "Einstein introduced general relativity in the year",
    "Einstein's theory of general relativity was published in",
    "The general theory of relativity was first published in",
    "General relativity was introduced by Einstein in",
    "Einstein first presented general relativity in",
    "The theory of general relativity dates from",
    "Einstein's paper on general relativity appeared in",
    "The general theory of relativity appeared in the year",
    "Einstein proposed general relativity in",
    "General relativity was proposed in the year",
    "The publication of general relativity came in",
    "Einstein's general theory of relativity emerged in",
    "The famous paper on general relativity was published in",
    "Gravity was reimagined when general relativity appeared in",
    "Einstein wrote his general relativity paper in",
    "The general relativity paper was released in",
    "Einstein developed general relativity in the year",
    "His theory of general relativity was announced in",
    "The theory we call general relativity arrived in",
    "Einstein completed his general theory of relativity in",
    "General relativity, published in",
    "The year Einstein published general relativity was",
    "Einstein unveiled general relativity in",
    "Our modern theory of gravity, general relativity, arrived in",
]
EIN = [
    "The theory of special relativity was proposed by",
    "The theory of relativity was developed by",
    "Special relativity was formulated by",
    "General relativity was formulated by",
    "The theory of relativity was created by",
    "Relativity theory was first proposed by",
    "The special theory of relativity was conceived by",
    "The general theory of relativity was conceived by",
    "The theory of relativity is credited to",
    "Special relativity is attributed to",
    "The physicist who proposed special relativity was",
    "The physicist who developed general relativity was",
    "The scientist behind the theory of relativity was",
    "E = mc2 was derived by",
    "Mass-energy equivalence was discovered by",
    "The curvature of spacetime was described by",
    "Time dilation was predicted by",
    "The relativity of simultaneity was introduced by",
    "The equivalence principle was proposed by",
    "The field equations of general relativity were written down by",
    "The photoelectric effect was explained by",
    "The annus mirabilis papers of 1905 were written by",
    "Brownian motion was explained in 1905 by",
    "The famous equation relating energy and mass came from",
]
BEHAVIOURS = [("kb0_sr1905", SR, " 1905"),
              ("kb1_gr1915", GR, " 1915"),
              ("kb2_einstein", EIN, " Einstein")]


def main():
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    tok = Tokenizer()
    inference.disable_compile()

    all_wins, all_tgts, all_probs, all_assign = [], [], [], []
    ok = True
    for bi, (name, prompts, cont) in enumerate(BEHAVIOURS):
        # candidates[(pi, j)] = (window63, target, decoded)
        cands, rows = {}, []
        for pi, pr in enumerate(prompts):
            pt = tok.encode(pr)
            full = tok.encode(pr + cont)
            if full[:len(pt)] != pt:
                print("  [%s] merge mismatch, skipped: %r" % (name, pr))
                continue
            ct = full[len(pt):]
            for j in range(len(ct)):
                seq = pt + ct[:j]
                if len(seq) > ANCHOR + 1:
                    continue
                win = [0] * (ANCHOR + 1 - len(seq)) + seq
                cands[(pi, j)] = (win, ct[j])
        keys = sorted(cands)
        p_of = {}
        with torch.no_grad():
            for s in range(0, len(keys), 16):
                ks = keys[s:s + 16]
                tk = torch.tensor([cands[k][0] for k in ks],
                                  dtype=torch.long, device=device)
                logits, _ = inference.model(tk)
                if logits.dim() == 3:
                    logits = logits[:, -1, :]
                p = torch.softmax(logits.float(), -1)
                for r, k in enumerate(ks):
                    p_of[k] = float(p[r, cands[k][1]])
        by_j = {}
        for (pi, j), pv in p_of.items():
            by_j.setdefault(j, []).append((pi, pv))
        best_j, best = None, -1
        for j, lst in by_j.items():
            kept = sum(1 for _, pv in lst if pv >= PKEEP)
            print("  [%s] split j=%d: %d/%d pass p>=%.2f (median p %.2f)"
                  % (name, j, kept, len(lst), PKEEP,
                     sorted(pv for _, pv in lst)[len(lst) // 2]))
            if kept > best:
                best, best_j = kept, j
        kept_pi = [pi for pi, pv in by_j[best_j] if pv >= PKEEP]
        tgt_tok = None
        for pi in kept_pi:
            win, t = cands[(pi, best_j)]
            all_wins.append(win + [t])
            all_tgts.append(t)
            all_probs.append(p_of[(pi, best_j)])
            all_assign.append(bi)
            tgt_tok = t
        print("== %s: split j=%d, kept %d/%d, target token %r"
              % (name, best_j, len(kept_pi), len(prompts),
                 tok.decode([tgt_tok]) if tgt_tok is not None else "?"),
              flush=True)
        if len(kept_pi) < MINKEEP:
            print("   !! below MINKEEP=%d" % MINKEEP)
            ok = False

    nb = len(BEHAVIOURS)
    assign = torch.tensor(all_assign)
    torch.save({"windows": all_wins, "targets": all_tgts,
                "probs": all_probs, "assign": assign,
                "coherence": torch.ones(nb),
                "sizes": torch.bincount(assign, minlength=nb),
                "anchor": ANCHOR}, HERE / "knowledge_clusters.pt")
    print("-> knowledge_clusters.pt (%d contexts)" % len(all_wins))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
