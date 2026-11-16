"""119 #18: TuringLLM SAE dCE under the public-model protocol (070 recon_audit.py).

Text: 070 recon_audit.text_tokens exactly (wikitext-103-raw-v1 train shard 0, the rows of the first 6,000 with
len > 200, joined), tokenised with TuringLLM's tokenizer (families.py tokenizer_id) WITHOUT special tokens (no BOS),
N_SEQ = 16 windows of SEQ = 64 = 1,024 tokens. Per site (36 = 12 layers x attn/mlp/resid):
  dce      CE(site replaced by its SAE reconstruction) - clean CE   (src/eval/sae/reconstruction_eval._SiteSplice
           "recon", the same splice the model card's TuringLLM column came from)
  zero     CE(site zeroed) - clean CE                                (_SiteSplice "zero")
  rec_pct  100 (zero - dce) / zero  (the old "loss recovered", for reference only)
  fvu      1 - explained variance of the reconstruction (070's formula: residual SS / SS about the global mean)
CPU, float32 (as 070 ran the public models).

  wsl: cd <repo>; PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= \
       .venv/bin/python -X utf8 experiments/119-strict-heldout-and-tl-dce/tl_dce.py
Writes out/tl_dce.json and out/tl_dce.csv.
"""
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
N_SEQ, SEQ = 16, 64
BOS = os.environ.get("BOS") == "1"      # env BOS=1: sensitivity arm (<s> + the same 64 tokens; writes tl_dce_bos.*)
SUF = "_bos" if BOS else ""
DEV = torch.device("cpu")
torch.set_grad_enabled(False)


def text_tokens(tok):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    pqf = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                          filename="wikitext-103-raw-v1/train-00000-of-00002.parquet")
    text = "".join(t for t in pq.read_table(pqf).column("text").to_pylist()[:6000] if len(t) > 200)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    assert ids.numel() >= N_SEQ * SEQ
    w = torch.stack([ids[i * SEQ:(i + 1) * SEQ] for i in range(N_SEQ)])
    if BOS:              # sensitivity arm: TuringLLM's training documents start with <s>; prepend it to each window
        w = torch.cat([torch.full((N_SEQ, 1), int(tok.bos_token_id), dtype=w.dtype), w], dim=1)
    return w


def ce(inference, tokens, patcher=None):
    _, logits, _ = inference.forward(tokens, num_gen=1, tokenize_final=False, return_activations=False,
                                     all_logits=True, patcher=patcher)
    return float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tokens[:, 1:].reshape(-1)))


def main():
    from transformers import AutoTokenizer
    from eval.sae.reconstruction_eval import _SiteSplice
    from model.families import FAMILIES, TURINGLLM
    from model.inference import Inference
    from sae.bank import SAEBank
    from sae.dense import sparse_topk_to_dense

    fam = FAMILIES[TURINGLLM]
    inference = Inference(DEV, compile=False)
    bank = SAEBank(device=DEV)
    tok = AutoTokenizer.from_pretrained(fam.tokenizer_id)
    tokens = text_tokens(tok).to(DEV)
    assert bool((tokens[:, 1 if BOS else 0:] != tok.bos_token_id).all())
    base = ce(inference, tokens)
    print("TuringLLM | %d x %d wikitext tokens (%s) | clean CE %.4f" % (
        N_SEQ, SEQ, "<s> prepended" if BOS else "no BOS", base), flush=True)

    # site inputs for FVU, from one clean pass
    _, _, acts = inference.forward(tokens, num_gen=1, tokenize_final=False, return_activations=True)
    rows = []
    for layer in range(fam.n_layer):
        for kidx, kind in enumerate(fam.kinds):
            x = acts[:, layer, kidx, 0].float()
            ta, ti = bank.encode(x, kind, layer)
            xh = bank.decode(sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype), kind, layer)
            fvu = float(((x - xh) ** 2).sum() / ((x - x.mean((0, 1))) ** 2).sum())
            c_rec = ce(inference, tokens, _SiteSplice(bank, layer, kind, "recon"))
            c_zero = ce(inference, tokens, _SiteSplice(bank, layer, kind, "zero"))
            dce, zero = c_rec - base, c_zero - base
            r = dict(kind=kind, layer=layer, fvu=fvu, dce=dce, zero=zero,
                     rec_pct=100.0 * (zero - dce) / zero if zero > 0 else None)
            rows.append(r)
            print("%-6s L%-2d FVU %.4f  dCE %+.4f  zero %+.4f  rec %s" % (
                kind, layer, fvu, dce, zero, "%.1f%%" % r["rec_pct"] if r["rec_pct"] is not None else "-"),
                flush=True)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    json.dump(dict(model="turingllm", tokenizer=fam.tokenizer_id, clean_ce=base, n_seq=N_SEQ, seq=SEQ, bos=BOS,
                   device="cpu", dtype="float32", rows=rows), open(out / ("tl_dce%s.json" % SUF), "w"), indent=1)
    with open(out / ("tl_dce%s.csv" % SUF), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote out/tl_dce%s.json, out/tl_dce%s.csv" % (SUF, SUF), flush=True)


if __name__ == "__main__":
    main()
