"""STRING-DETECTOR CONTROL for the running example (3.resid.35381): a concept latent or a "temperature" token detector?

Its top contexts peak on the token "temperature" 61/64 times (95%), all in physics / thermodynamics passages. On a
RANDOM corpus sample this asks:
  - recall: on what share of "temperature"-family tokens (temperature, temper + atures, ...) does it fire?
  - precision: what share of its firings are on those tokens?
  - does it separate senses? Windows around "temperature" tokens where it fires strongly vs where it stays silent,
    so the contexts can be read (physics / chemistry vs weather / cooking / body temperature).
A token detector fires on every "temperature" regardless of sense; a concept latent fires selectively.

  PYTHONPATH=src python experiments/060-running-example-v1/string_control.py  -> results/string_control.md
  env: N_SEQ (16384 random sequences)  SEED (0)
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "049-circuit-graph"))
N_SEQ = int(os.environ.get("N_SEQ", 16384))
RNG = np.random.default_rng(int(os.environ.get("SEED", 0)))
TARGET = (3, "resid", 35381)
BS = 64


def main():
    os.environ.setdefault("TAG", "strctl_unused")
    import amp_eval_pass_v2 as V
    from model.tokenizer import Tokenizer
    from sae.dense import target_latent_activations
    G = V.setup()
    inference, bank, M0, KINDS = (G[k] for k in ("inference", "bank", "M0", "KINDS"))
    tok = Tokenizer()
    sel = M0._neg_context_selector(); loader = sel.loader
    vocab = int(inference.model.lm_head.weight.shape[0])
    dec = lambda ids: tok.decode(ids).replace("\n", "\\n")
    # the word itself, or its subword split "temper" + "ature(s)" (a bare "atures" also ends "creatures", "mature")
    whole = sorted(t for t in range(vocab) if tok.decode([t]).strip().lower() in ("temperature", "temperatures"))
    head = sorted(t for t in range(vocab) if tok.decode([t]).strip().lower() == "temper")
    tail = sorted(t for t in range(vocab) if tok.decode([t]).strip().lower() in ("atures", "ature"))
    fam = whole + tail
    print("temperature tokens:", [(t, tok.decode([t])) for t in whole], "split:", head, tail, flush=True)
    ranges = loader.shard_id_ranges
    sizes = np.array([e - s + 1 for s, e in ranges], dtype=np.float64)
    shards = RNG.choice(len(ranges), size=N_SEQ, p=sizes / sizes.sum())
    ids = sorted({int(ranges[s][0] + RNG.integers(0, sizes[s])) for s in shards})
    ids, tokens = sel.load_tokens(ids, max_length=64)
    l, k, i = TARGET
    acts = []

    def hook(layer_idx, activations):
        if layer_idx == l:
            ta, ti = bank.encode(activations[KINDS.index(k)], k, layer_idx)
            acts.append(target_latent_activations(ta, ti, i).float().cpu())
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), BS):
                inference.forward(tokens[s0:s0 + BS], activations_callback=hook, return_activations=False,
                                  tokenize_final=False)
    finally:
        inference.enable_compile()
    A = torch.cat(acts)                                                  # [N, 64]
    T = tokens.cpu()
    prev = torch.cat([torch.full((T.shape[0], 1), -1, dtype=T.dtype), T[:, :-1]], 1)
    is_fam = torch.isin(T, torch.tensor(whole)) | (torch.isin(T, torch.tensor(tail)) & torch.isin(prev, torch.tensor(head)))
    on = A > 0
    n_fam = int(is_fam.sum()); n_on = int(on.sum())
    lines = ["# String-detector control: %d.%s.%d on %d random sequences (%d positions)\n" % (l, k, i, T.shape[0], T.numel()),
             "- temperature-family tokens: %d; target fires on %d of them (recall %.2f)" % (n_fam, int((on & is_fam).sum()),
                                                                                           (on & is_fam).sum().item() / max(1, n_fam)),
             "- target fires on %d positions; %d of them are temperature-family tokens (precision %.2f)" % (
                 n_on, int((on & is_fam).sum()), (on & is_fam).sum().item() / max(1, n_on)),
             "- activation on family tokens where it fires: median %.2f, max %.2f" % (
                 float(A[on & is_fam].median()) if (on & is_fam).any() else 0, float(A[on & is_fam].max()) if (on & is_fam).any() else 0)]
    top_other = torch.bincount(T[on & ~is_fam], minlength=vocab).topk(8)
    lines.append("- its other firings, top tokens: %s" % ", ".join("%r×%d" % (dec([int(t)]), int(c)) for c, t in
                                                                   zip(top_other.values, top_other.indices) if c > 0))

    def windows(mask, n, strongest=True):
        pos = mask.nonzero().tolist()
        vals = [float(A[b, p]) for b, p in pos]
        order = np.argsort(vals)[::-1] if strongest else RNG.permutation(len(pos))
        out = []
        for j in order[:n]:
            b, p = pos[j]
            out.append("%.2f | …%s [[%s]]%s" % (float(A[b, p]), dec(T[b, max(0, p - 14):p].tolist()), dec([int(T[b, p])]),
                                                dec(T[b, p + 1:p + 5].tolist())))
        return out
    lines.append("\n## Temperature-family tokens where the target fires most strongly\n")
    lines += ["- " + w for w in windows(on & is_fam, 12)]
    lines.append("\n## Temperature-family tokens where the target stays SILENT (random sample)\n")
    lines += ["- " + w for w in windows(~on & is_fam, 12, strongest=False)]
    txt = "\n".join(lines)
    (HERE / "results" / "string_control.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
