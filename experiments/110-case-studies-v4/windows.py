"""CPU look at candidate targets: anchor windows of the strongest fresh-record contexts, anchor-token histogram,
contrast snippets, production scores. No model pass.

  PYTHONPATH=src python -X utf8 experiments/110-case-studies-v4/windows.py KEY [KEY ...]  (N_WIN, default 10)
"""
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib110 as L  # noqa: E402

N_WIN = int(os.environ.get("N_WIN", 10))


def main():
    from model.tokenizer import Tokenizer
    tok = Tokenizer()
    dec = lambda ids: tok.decode([int(t) for t in ids if int(t) >= 0]).replace("\n", " ")
    d = L.scores()
    for key in sys.argv[1:]:
        r = d.loc[key]
        rec = L.record(key)
        pos, arg, neg = rec["strong"]["pos"], rec["strong"]["arg"].long(), rec["neg"]
        print("\n#### %s | %s n=%d Z %.2f A %.2f C %.2f nec %.2f" % (
            key, "PASS" if r.passes else "fail", r.n, r.free0_tk, r.freeM_topk_tk or float("nan"),
            r.freeN_topk_tk or float("nan"), r.phi_sup_blind_tk))
        anc = Counter(dec([pos[b, int(arg[b])]]) for b in range(pos.shape[0]))
        print("anchors:", anc.most_common(8))
        seen = []
        for b in range(pos.shape[0]):
            a = int(arg[b])
            w = dec(pos[b, max(0, a - 22):a]) + " [[" + dec(pos[b, a:a + 1]) + "]]" + dec(pos[b, a + 1:a + 4])
            if w not in seen:
                seen.append(w)
            if len(seen) >= N_WIN:
                break
        for w in seen:
            print("  -", w)
        for b in (0, 1, 2):
            print("  contrast:", dec(neg[b, 10:40]))


if __name__ == "__main__":
    main()
