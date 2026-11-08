"""PART B SCREEN (CPU): which production TuringLLM targets are about physics, AI / machine learning, or computing?

Token-driven, no auto-interp. For every fitted target, decode its fresh-record contexts and count domain terms:
  - strongest pool (64 contexts): share of contexts whose text contains a domain term (the whole 64-token window),
  - contrast pool (64 contexts, target silent): the same share, so a target whose contrast contexts are ALSO about the
    domain (a sub-concept inside the domain) is visible, and one that fires on everything in a domain document is not
    confused with one that fires on any document,
  - anchor-token check (string-detector screen): share of strongest contexts whose ANCHOR token is itself a domain
    term (a "physics" latent that peaks on the token "physics"), and peak-token consistency (share on the most common
    anchor token).
Joined with the production scores (pass rule, size). Writes out/topic_scan.csv; prints the top of each domain.

  PYTHONPATH=src python -X utf8 experiments/110-case-studies-v4/topic_scan.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib110 as L  # noqa: E402

TERMS = {
    "physics": r"physic|quantum|relativ|electron|photon|neutron|proton|particle|momentum|velocit|gravit|"
               r"electromagnet|magnetic field|thermodynam|entropy|spacetime|space-time|cosmolog|black hole|"
               r"newton|einstein|maxwell|kinetic|wavelength|nuclear|atomic|subatomic|boson|quark|optics|"
               r"acceleration|inertia|superconduct|semiconductor|voltage|electric field|planck|schr[oö]dinger|"
               r"heisenberg|big bang|dark matter|string theory|fusion|fission|radiation|laser",
    "ai": r"machine learning|artificial intelligence|neural net|deep learning|reinforcement learning|"
          r"supervised|unsupervised|backprop|gradient descent|natural language processing|\bnlp\b|"
          r"computer vision|\bai\b|training data|classifier|transformer model|large language model|\bllm|"
          r"chatbot|robot|autonomous|intelligent system|expert system|turing test|perceptron|convolutional",
    "computing": r"computer|software|programming|program code|source code|\bcode\b|coding|python|javascript|"
                 r"\bjava\b|\bc\+\+|algorithm|data structure|database|\bsql\b|compiler|\bcpu\b|processor|"
                 r"operating system|internet|cybersecurity|encryption|cryptograph|binary|\bbyte|\bbit\b|"
                 r"\bhtml|server|cloud computing|hardware|\bdebug|variable|function call|recursion|"
                 r"programmer|developer|\bapi\b|silicon|transistor|computation|computing|digital",
}
RX = {d: re.compile(p, re.I) for d, p in TERMS.items()}


def main():
    from model.tokenizer import Tokenizer
    tok = Tokenizer()
    d = L.scores("strong")
    rows = []
    for n, key in enumerate(d.index):
        p = L.CTX_FRESH / ("%s.pt" % key)
        if not p.exists():
            continue
        rec = torch.load(p, weights_only=False)[key]
        pos, arg, neg = rec["strong"]["pos"], rec["strong"]["arg"].long(), rec["neg"]
        ptxt = [tok.decode([int(t) for t in r if int(t) >= 0]) for r in pos]
        ntxt = [tok.decode([int(t) for t in r if int(t) >= 0]) for r in neg]
        anc = [tok.decode([int(pos[b, int(arg[b])])]).strip() for b in range(pos.shape[0])]
        # the anchor's word: the anchor token plus its neighbours, so a subword piece of "physics" counts
        aw = [tok.decode([int(t) for t in pos[b, max(0, int(arg[b]) - 1):int(arg[b]) + 2]])
              for b in range(pos.shape[0])]
        cons = Counter(anc).most_common(1)[0]
        row = dict(seed=key, consistency=cons[1] / len(anc), top_anchor=cons[0])
        for dom, rx in RX.items():
            row[dom + "_pos"] = sum(bool(rx.search(t)) for t in ptxt) / len(ptxt)
            row[dom + "_neg"] = sum(bool(rx.search(t)) for t in ntxt) / max(1, len(ntxt))
            row[dom + "_anchor"] = sum(bool(rx.search(t)) for t in aw) / len(aw)
        rows.append(row)
        if n % 500 == 0:
            print(n, key, flush=True)
    df = pd.DataFrame(rows).set_index("seed")
    df = df.join(d[["layer", "kind", "n", "passes", "vacuous", "worst_dev", "free0_tk", "freeM_topk_tk",
                    "freeN_topk_tk", "phi_sup_blind_tk"]])
    dm = L.scores("mid")
    df["mid_passes"] = dm.passes.reindex(df.index)
    L.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(L.RESULTS / "topic_scan.csv")
    for dom in RX:
        s = df[df.passes].sort_values(dom + "_pos", ascending=False)
        print("\n== %s: %d passing targets with >= 80%% of strongest contexts on-domain" % (
            dom, int((s[dom + "_pos"] >= 0.8).sum())))
        print(s[[dom + "_pos", dom + "_neg", dom + "_anchor", "consistency", "top_anchor", "n", "layer", "kind"]]
              .head(30).to_string())


if __name__ == "__main__":
    main()
