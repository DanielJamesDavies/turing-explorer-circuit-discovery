"""LOGIT LENS on fact completions: project each layer's residual
stream through norm_f + the tied unembedding and track the answer
token's probability and rank, layer by layer, at the answer position.

Shows WHERE in depth the completion crystallises -- read alongside the
year-ladder (L4-L9) to see whether the answer's emergence tracks the
relay's rungs.

  PYTHONPATH=src python .../logit_lens.py
"""
import sys

sys.path.insert(0, "src")
import torch

from hardware import detect_devices, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer

PROBES = [
    "Einstein published his theory of special relativity in the year",
    "The theory of relativity was developed by Albert",
    "General relativity describes gravity not as a force but as the curvature of",
]


def main():
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    tok = Tokenizer()
    model = inference.model
    inference.disable_compile()
    norm_f = model.transformer.norm_f
    W_U = model.lm_head.weight            # [vocab, d], tied with wte

    for prompt in PROBES:
        ids = [1] + [i for i in tok.encode(prompt) if i != 1]
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        resids = {}

        def cb(layer_idx, acts):
            resids[layer_idx] = acts[2][:, -1, :].detach()   # resid, last pos

        with torch.no_grad():
            inference.forward(idx, num_gen=1, tokenize_final=False,
                              activations_callback=cb,
                              return_activations=False)
            logits, _ = model(idx)
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            ans = int(logits.argmax(-1))
        ans_txt = tok.decode([ans]) or repr(ans)
        print("\n=== %r -> %r" % (prompt[-46:], ans_txt))
        print("%-6s %10s %7s   %s" % ("layer", "P(ans)", "rank", "top-3"))
        with torch.no_grad():
            for L in sorted(resids):
                h = norm_f(resids[L].to(W_U.dtype))
                lg = (h @ W_U.T).float()
                p = torch.softmax(lg, -1)[0]
                rank = int((p > p[ans]).sum()) + 1
                top3 = [tok.decode([int(t)]) or "?"
                        for t in lg.topk(3).indices[0].tolist()]
                print("L%-5d %10.5f %7d   %s"
                      % (L, float(p[ans]), rank,
                         " | ".join(repr(t) for t in top3)))
    inference.enable_compile()


if __name__ == "__main__":
    main()
