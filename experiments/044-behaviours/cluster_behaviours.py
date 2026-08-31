"""UNSUPERVISED BEHAVIOUR DISCOVERY, after Marks et al. (Sparse Feature
Circuits, ICLR 2025) who use the quanta-clustering approach of Michaud
et al. 2023: cluster contexts "based on vectors derived from
activations, gradients or both"; a cluster of (context, next-token)
pairs defines a BEHAVIOUR with metric -log P(y|x). This is the
ACTIVATIONS variant (one of the paper's stated options).

Pipeline here (TuringLLM):
  1. sample training windows (proper -1-delimited segmentation);
     prediction task = final position (62 -> token 63);
  2. keep contexts the model predicts CONFIDENTLY (p > PMIN) --
     behaviours are things the model can do;
  3. representation = concat over layers of the L2-normalised residual
     stream at the prediction position (12 x 1024, renormalised);
  4. spherical k-means, K clusters;
  5. report cluster sizes + coherence; decode samples of the most
     coherent clusters; save everything to behaviour_clusters.pt for
     the tri-amp (objective="logit") stage.

  PYTHONPATH=src python experiments/044-behaviours/cluster_behaviours.py
  env: NWIN (8192), K (100), PMIN (0.2)
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from hardware import detect_devices, should_compile
from model.inference import Inference
from model.tokenizer import Tokenizer

HERE = Path(__file__).parent
SEQ = 64
NWIN = int(os.environ.get("NWIN", 8192))
K = int(os.environ.get("K", 100))
PMIN = float(os.environ.get("PMIN", 0.2))
BATCH = 32
ANCHOR = SEQ - 2                     # predict the final token


def sample_windows(n):
    shards = sorted(Path("data").glob("*.npy"))
    rng = np.random.default_rng(17)
    out = []
    for sp in rng.choice(shards, size=min(400, len(shards)), replace=False):
        sh = np.asarray(np.load(sp, mmap_mode="r"))
        sep = np.where(sh == -1)[0]
        st = np.concatenate([[0], sep + 1]) + 1
        en = np.concatenate([sep, [len(sh)]])
        keep = (en - st) == SEQ
        st, en = st[keep], en[keep]
        for a, b in zip(st[: n // 350 + 1], en[: n // 350 + 1]):
            out.append(sh[a:b].tolist())
        if len(out) >= n:
            break
    return out[:n]


def main():
    devices = detect_devices()
    device = devices[0]
    inference = Inference(device=device, compile=should_compile())
    tok = Tokenizer()
    model = inference.model
    wins = sample_windows(NWIN)
    print("%d windows sampled" % len(wins), flush=True)

    reps, kept, targets, probs = [], [], [], []
    inference.disable_compile()
    for s0 in range(0, len(wins), BATCH):
        chunk = wins[s0:s0 + BATCH]
        toks = torch.tensor([[max(t, 0) for t in w] for w in chunk],
                            dtype=torch.long, device=device)
        resids = {}

        def cb(layer_idx, acts):
            resids[layer_idx] = acts[2][:, ANCHOR, :].detach().float()

        with torch.no_grad():
            inference.forward(toks[:, :ANCHOR + 1], num_gen=1,
                              tokenize_final=False,
                              activations_callback=cb,
                              return_activations=False)
            logits, _ = model(toks[:, :ANCHOR + 1])
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            p = torch.softmax(logits.float(), -1)
        for b, w in enumerate(chunk):
            y = w[ANCHOR + 1]
            if y < 0:
                continue
            py = float(p[b, y])
            if py < PMIN:
                continue
            v = torch.cat([torch.nn.functional.normalize(
                resids[L][b], dim=0) for L in sorted(resids)])
            reps.append(torch.nn.functional.normalize(v, dim=0).cpu())
            kept.append(w)
            targets.append(y)
            probs.append(py)
        if (s0 // BATCH) % 40 == 0:
            print("  %d/%d scanned, %d kept" % (s0 + len(chunk), len(wins),
                                                len(kept)), flush=True)
    inference.enable_compile()
    X = torch.stack(reps).to(device)
    print("kept %d confident contexts (p >= %.2f); clustering K=%d"
          % (len(kept), PMIN, K), flush=True)

    # spherical k-means
    g = torch.Generator(device="cpu").manual_seed(5)
    idx0 = torch.randperm(X.shape[0], generator=g)[:K]
    C = X[idx0.to(X.device)].clone()
    for it in range(50):
        sims = X @ C.T
        assign = sims.argmax(1)
        newC = torch.zeros_like(C)
        for k in range(K):
            m = assign == k
            if int(m.sum()) > 0:
                newC[k] = torch.nn.functional.normalize(X[m].mean(0), dim=0)
            else:
                newC[k] = C[k]
        if torch.allclose(newC, C, atol=1e-5):
            C = newC
            break
        C = newC
    sims = X @ C.T
    assign = sims.argmax(1).cpu()
    coh = torch.zeros(K)
    for k in range(K):
        m = assign == k
        if int(m.sum()) > 1:
            coh[k] = sims[m.to(sims.device), k].mean().cpu()

    sizes = torch.bincount(assign, minlength=K)
    order = torch.argsort(coh, descending=True)
    print("\ntop clusters by coherence:")
    for k in order[:12].tolist():
        if sizes[k] < 24:
            continue
        members = (assign == k).nonzero(as_tuple=True)[0].tolist()
        print("\n== cluster %d | size %d | coherence %.3f" %
              (k, int(sizes[k]), float(coh[k])))
        for i in members[:3]:
            w = kept[i]
            txt = tok.decode([t for t in w[max(0, ANCHOR - 14):ANCHOR + 1]
                              if t >= 0])
            print("   ...%s -> %r (p=%.2f)" %
                  (txt.replace(chr(10), " ")[-90:],
                   tok.decode([targets[i]]), probs[i]))

    torch.save({"windows": kept, "targets": targets, "probs": probs,
                "assign": assign, "coherence": coh, "sizes": sizes,
                "anchor": ANCHOR}, HERE / "behaviour_clusters.pt")
    print("\n-> behaviour_clusters.pt")


if __name__ == "__main__":
    main()
