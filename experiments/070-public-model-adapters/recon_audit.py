"""RECONSTRUCTION AUDIT through the adapters (DAN-35 done-criterion).

Per site (kind x layer), on the text 053's sae_recon.py used (wikitext-103 train
rows with len > 200 from the first 6,000, joined, tokenised without BOS, N_SEQ
windows of SEQ tokens):

  fvu      1 - explained variance of the reconstruction, model space (053's formula)
  slots    Top-K slots per token (must be exactly k)
  l0       non-zeros per token (= k on Pythia; <= k on GPT-2, where SAELens' topk-then-ReLU zeros a few)
  dce      CE increase when the site is REPLACED by its reconstruction, through the adapter's edit path
  ident    max |logit change| for a zero decoded delta through the same path (must be 0)
  ref      max |code - reference code|: the bank's dense code vs an independent transcription
           (Pythia: 033's crosssae_topk formulas on the raw safetensors; GPT-2: 053's gpt2saes module)
  GPT-2 only:
  dce_053bug  resid_mid replaced the way 053 did it (pre-hook on ln_2 = MLP input only; DAN-68)

  wsl: PYTHONPATH=src CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
       .venv/bin/python experiments/070-public-model-adapters/recon_audit.py
  env: FAMILY (pythia-70m | gpt2-small)  N_SEQ (16)  SEQ (64)  LAYERS (all)
Writes recon_<family>.json next to this file. Sites whose SAE is not in the local
HF cache are listed as missing.
"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
EXP = HERE.parent
FAMILY = os.environ.get("FAMILY", "pythia-70m")
N_SEQ = int(os.environ.get("N_SEQ", 16))
SEQ = int(os.environ.get("SEQ", 64))
DEV = torch.device("cpu")
torch.set_grad_enabled(False)


def text_tokens(tok):
    """053/sae_recon.py's text, exactly."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    pqf = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                          filename="wikitext-103-raw-v1/train-00000-of-00002.parquet")
    text = "".join(t for t in pq.read_table(pqf).column("text").to_pylist()[:6000] if len(t) > 200)
    ids = tok(text, return_tensors="pt").input_ids[0]
    return torch.stack([ids[i * SEQ:(i + 1) * SEQ] for i in range(N_SEQ)])


def hook_point(hf, kind, layer):
    if FAMILY == "pythia-70m":
        lay = hf.gpt_neox.layers[layer]
        return {"attn": (lay.attention, "out"), "mlp": (lay.mlp, "out"), "resid": (lay, "out")}[kind]
    b = hf.transformer.h[layer]
    return {"attn": (b.attn, "out"), "resid_mid": (b.ln_2, "in"), "mlp": (b.mlp, "out"), "resid": (b, "out")}[kind]


def raw_stream(hf, tokens, kind, layer):
    """The model-space stream at the hook point, from a plain HF hook (independent of the adapter)."""
    mod, mode = hook_point(hf, kind, layer)
    cap = {}
    if mode == "in":
        h = mod.register_forward_pre_hook(lambda m, a: cap.__setitem__("x", a[0]))
    else:
        h = mod.register_forward_hook(lambda m, a, o: cap.__setitem__("x", o[0] if isinstance(o, tuple) else o))
    try:
        hf(input_ids=tokens, use_cache=False)
    finally:
        h.remove()
    return cap["x"]


def ce_with(inference, tokens, transform=None):
    from model.hooks import multi_patch
    ctx = multi_patch(inference.model, transform) if transform else None
    if ctx:
        ctx.__enter__()
    try:
        logits, _ = inference.model(tokens, return_all_logits=True)
    finally:
        if ctx:
            ctx.__exit__(None, None, None)
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)).item(), logits


def reference_code(kind, layer, x_raw, bank):
    """Independent transcription -> dense code [.., d_sae]."""
    if FAMILY == "pythia-70m":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        hp = "layers.%d" % layer if kind == "resid" else "layers.%d.%s" % (layer, "attention" if kind == "attn" else "mlp")
        sd = load_file(hf_hub_download("EleutherAI/sae-pythia-70m-32k", "%s/sae.safetensors" % hp))
        pre = (x_raw - sd["b_dec"]) @ sd["encoder.weight"].T + sd["encoder.bias"]      # 033 pre_acts
        p = F.relu(pre)
        v, i = p.topk(16, dim=-1)
        return torch.zeros_like(p).scatter(-1, i, v)
    sys.path.insert(0, str(EXP / "053-gpt2-topk"))
    import gpt2saes as G
    t = G.load_sae({"attn": "attn-out", "resid_mid": "resid-mid", "mlp": "mlp-out", "resid": "resid-post"}[kind],
                   layer, device="cpu")
    xn, _, _ = G.norm_in(x_raw)
    return G.encode(t, xn)


def main():
    from model.families import FAMILIES
    from model.inference import Inference
    from sae.bank import SAEBank
    from sae.dense import sparse_topk_to_dense
    from transformers import AutoTokenizer

    fam = FAMILIES[FAMILY]
    inference = Inference(DEV, family=FAMILY)
    bank = SAEBank(device=DEV, family=FAMILY, preload=False)
    hf = inference.model.hf
    tok = AutoTokenizer.from_pretrained(fam.tokenizer_id)
    tokens = text_tokens(tok)
    base, base_logits = ce_with(inference, tokens)
    print("%s | %d x %d wikitext tokens | clean CE %.4f\n" % (FAMILY, N_SEQ, SEQ, base), flush=True)
    layers = ([int(x) for x in os.environ["LAYERS"].split(",")] if os.environ.get("LAYERS")
              else list(range(fam.n_layer)))
    rows, missing = [], []
    hdr = "%-10s %-3s %8s %6s %6s %9s %8s %9s" % ("kind", "L", "FVU", "slots", "L0", "dCE", "ident", "ref")
    print(hdr + ("  dCE_053bug" if FAMILY == "gpt2-small" else ""), flush=True)
    for layer in layers:
        for kind in fam.kinds:
            from huggingface_hub import try_to_load_from_cache
            repo, wfile, _ = fam.sae.files(kind, layer)
            if not isinstance(try_to_load_from_cache(repo, wfile), str):
                missing.append("%s L%d" % (kind, layer))
                print("%-10s %-3d MISSING (not in HF cache: %s/%s)" % (kind, layer, repo, wfile), flush=True)
                continue
            x_raw = raw_stream(hf, tokens, kind, layer)
            x = bank.site_input(x_raw, kind, layer)
            ta, ti = bank.encode(x, kind, layer)
            code = sparse_topk_to_dense(ta, ti, bank.d_sae)
            x_hat_s = bank.decode(code, kind, layer)
            scale = (x_raw - x_raw.mean(-1, keepdim=True)).std(-1, keepdim=True) + bank.norm_eps \
                if bank.normalize == "layer_norm" else None
            err_model = (x - x_hat_s) * (scale if scale is not None else 1.0)
            fvu = float((err_model ** 2).sum() / ((x_raw - x_raw.mean((0, 1))) ** 2).sum())
            l0 = float((ta > 0).sum(-1).float().mean())
            slots = int(ti.shape[-1])
            ref = float((code - reference_code(kind, layer, x_raw, bank)).abs().max())

            def recon(l, k, xs, _s=(layer, kind)):
                if (l, k) != _s:
                    return None
                ta_, ti_ = bank.encode(xs, k, l)
                return bank.decode(sparse_topk_to_dense(ta_, ti_, bank.d_sae), k, l)

            zero = torch.zeros(bank.d_sae)

            def ident(l, k, xs, _s=(layer, kind)):
                if (l, k) != _s:
                    return None
                return xs + bank.decode(zero.expand(*xs.shape[:-1], -1), k, l, add_bias=False)

            ce, _ = ce_with(inference, tokens, recon)
            _, lg = ce_with(inference, tokens, ident)
            r = dict(kind=kind, layer=layer, fvu=fvu, slots=slots, l0=l0, dce=ce - base,
                     ident=float((lg - base_logits).abs().max()), ref_max_abs=ref)
            line = "%-10s %-3d %8.4f %6d %6.2f %+9.4f %8.1e %9.1e" % (kind, layer, fvu, slots, l0, ce - base,
                                                                   r["ident"], ref)
            if FAMILY == "gpt2-small" and kind == "resid_mid":
                # 053's splice: replace the ln_2 INPUT only (the block keeps its residual) — DAN-68
                ln2 = hf.transformer.h[layer].ln_2
                h = ln2.register_forward_pre_hook(
                    lambda m, a: ((a[0] - a[0].mean(-1, keepdim=True)).std(-1, keepdim=True) + bank.norm_eps)
                    * recon(layer, "resid_mid", bank.site_input(a[0], "resid_mid", layer)) + a[0].mean(-1, keepdim=True))
                try:
                    ce_bug, _ = ce_with(inference, tokens)
                finally:
                    h.remove()
                r["dce_053bug"] = ce_bug - base
                line += "  %+9.4f" % r["dce_053bug"]
            print(line, flush=True)
            rows.append(r)
            bank.release(kind, layer)
    out = HERE / ("recon_%s.json" % FAMILY)
    json.dump({"family": FAMILY, "clean_ce": base, "n_seq": N_SEQ, "seq": SEQ, "rows": rows, "missing": missing},
              open(out, "w"), indent=1)
    print("\nwrote %s%s" % (out.name, ("; MISSING: " + ", ".join(missing)) if missing else ""), flush=True)


if __name__ == "__main__":
    main()
