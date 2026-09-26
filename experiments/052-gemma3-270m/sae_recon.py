"""STEP 1 — RECONSTRUCTION AUDIT of Gemma Scope 2 (16k, L0 big) on Gemma 3 270M.

For every one of the 54 SAEs (18 layers x att/mlp/res), on wikitext-103 test
text (BOS excluded from every statistic and never spliced):
  EV        explained variance 1 - SSE / TSS, for BOTH encode conventions
            (plain, and x - b_dec before encoding); the convention whose L0
            matches the config is the right one
  L0        measured mean active latents per token vs the config's l0
  unseen    fraction of latents that never fired in the sample (an upper bound
            on dead latents — rare latents also land here)
  dCE       next-token loss with the SAE reconstruction spliced in, minus clean
  recovered (L_zero - L_splice) / (L_zero - L_clean), zero-ablating the site

This doubles as the weights check for the unsloth mirror: SAEs trained on
Google's weights only reconstruct well on the same weights.

  PYTHONPATH=src python experiments/052-gemma3-270m/sae_recon.py
Env: MODEL (unsloth/gemma-3-270m) | N_SEQ (48) | SEQ (256) | BS (8)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gemmascope2 as GS  # noqa: E402

MODEL = os.environ.get("MODEL", "unsloth/gemma-3-270m")
N_SEQ = int(os.environ.get("N_SEQ", 48)); SEQ = int(os.environ.get("SEQ", 256)); BS = int(os.environ.get("BS", 8))
DEV = torch.device("cuda")
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
LAYERS = model.model.layers
NL = len(LAYERS)
print("model %s | %d layers | d_model %d" % (MODEL, NL, model.config.get_text_config().hidden_size), flush=True)

# ---- text -------------------------------------------------------------------------------
pq = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
text = "\n".join(t for t in pd.read_parquet(pq)["text"].tolist() if t.strip())
ids = tok(text, add_special_tokens=False)["input_ids"]
bos = tok.bos_token_id
need = N_SEQ * (SEQ - 1)
assert len(ids) >= need, "not enough text"
chunks = [[bos] + ids[i * (SEQ - 1):(i + 1) * (SEQ - 1)] for i in range(N_SEQ)]
TOK = torch.tensor(chunks, dtype=torch.long)
print("text: wikitext-103 test, %d tokens available, using %d x %d (BOS prepended)" % (len(ids), N_SEQ, SEQ), flush=True)

# ---- SAEs -------------------------------------------------------------------------------
SITES = [(k, l) for l in range(NL) for k in ("att", "mlp", "res")]
t0 = time.time()
SAE = {s: GS.load_sae(s[0], s[1], device=DEV) for s in SITES}
CFG_L0 = {s: GS.config(s[0], s[1]).get("l0") for s in SITES}
print("loaded %d SAEs (%s, l0 %s) in %.0fs" % (len(SAE), GS.WIDTH, GS.L0, time.time() - t0), flush=True)


def module_for(site):
    k, l = site
    blk = LAYERS[l]
    return {"att": blk.self_attn.o_proj, "mlp": blk.post_feedforward_layernorm, "res": blk}[k]


class Hook:
    """capture (fn=None) or replace the site's tensor with fn(x) at positions 1: (BOS untouched)."""

    def __init__(self, site, fn=None):
        self.site, self.fn, self.out = site, fn, None
        mod = module_for(site)
        if site[0] == "att":
            self.h = mod.register_forward_pre_hook(self._pre)
        else:
            self.h = mod.register_forward_hook(self._post)

    def _apply(self, x):
        if self.fn is None:
            self.out = x.detach()
            return None
        y = x.clone()
        y[:, 1:] = self.fn(x[:, 1:])
        return y

    def _pre(self, mod, args):
        y = self._apply(args[0])
        return None if y is None else (y,) + tuple(args[1:])

    def _post(self, mod, args, out):
        x = out[0] if isinstance(out, tuple) else out
        y = self._apply(x)
        if y is None:
            return None
        return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y

    def remove(self):
        self.h.remove()


def token_loss(tk):
    """mean next-token CE over positions 1..T-1 predicted from 0..T-2, computed row-wise."""
    lg = model(tk).logits
    tot = 0.0
    for i in range(tk.shape[0]):
        tot += float(F.cross_entropy(lg[i, :-1].float(), tk[i, 1:], reduction="sum"))
    return tot, tk.shape[0] * (tk.shape[1] - 1)


# ---- pass 1: reconstruction statistics -----------------------------------------------------
acc = {s: dict(n=0, sse=[0.0, 0.0], sx=None, sxx=0.0, l0=[0.0, 0.0], fired=[None, None]) for s in SITES}
for b0 in range(0, N_SEQ, BS):
    tk = TOK[b0:b0 + BS].to(DEV)
    hooks = [Hook(s) for s in SITES]
    model.model(tk)
    for h in hooks:
        h.remove()
        s = h.site
        x = h.out[:, 1:].float().reshape(-1, h.out.shape[-1])
        a = acc[s]
        a["n"] += x.shape[0]
        a["sx"] = x.sum(0) if a["sx"] is None else a["sx"] + x.sum(0)
        a["sxx"] += float((x * x).sum())
        for v, sub in enumerate((False, True)):
            code = GS.encode(SAE[s], x, sub_bdec=sub)
            xh = GS.decode(SAE[s], code)
            a["sse"][v] += float(((x - xh) ** 2).sum())
            a["l0"][v] += float((code > 0).sum())
            f = (code > 0).any(0)
            a["fired"][v] = f if a["fired"][v] is None else (a["fired"][v] | f)
        del x

rec = {}
for s in SITES:
    a = acc[s]
    tss = a["sxx"] - float((a["sx"] * a["sx"]).sum()) / a["n"]
    rec[s] = dict(ev=[1 - a["sse"][v] / tss for v in (0, 1)], l0=[a["l0"][v] / a["n"] for v in (0, 1)],
                  unseen=[1 - float(a["fired"][v].float().mean()) for v in (0, 1)], cfg_l0=CFG_L0[s])
# pick the convention whose measured L0 is closer to the configured l0 (majority over sites)
votes = sum(abs(rec[s]["l0"][1] - rec[s]["cfg_l0"]) < abs(rec[s]["l0"][0] - rec[s]["cfg_l0"]) for s in SITES)
SUB = votes > len(SITES) / 2
print("\nencode convention: %s (sub_bdec closer to config l0 at %d/%d sites)"
      % ("x - b_dec" if SUB else "plain x", votes, len(SITES)), flush=True)
V = int(SUB)

# ---- pass 2: splice and zero-ablation loss -------------------------------------------------
clean_tot = clean_n = 0.0
for b0 in range(0, N_SEQ, BS):
    t_, n_ = token_loss(TOK[b0:b0 + BS].to(DEV)); clean_tot += t_; clean_n += n_
L_clean = clean_tot / clean_n
print("clean next-token loss %.4f nats" % L_clean, flush=True)
for s in SITES:
    tot = {"splice": 0.0, "zero": 0.0}
    for mode in ("splice", "zero"):
        if mode == "splice":
            fn = (lambda x, _s=s: GS.decode(SAE[_s], GS.encode(SAE[_s], x.float(), sub_bdec=SUB)).to(x.dtype))
        else:
            fn = (lambda x: torch.zeros_like(x))
        h = Hook(s, fn)
        for b0 in range(0, N_SEQ, BS):
            t_, _ = token_loss(TOK[b0:b0 + BS].to(DEV)); tot[mode] += t_
        h.remove()
    Ls, Lz = tot["splice"] / clean_n, tot["zero"] / clean_n
    rec[s].update(L_splice=Ls, L_zero=Lz, dCE=Ls - L_clean,
                  recovered=(Lz - Ls) / (Lz - L_clean) if abs(Lz - L_clean) > 1e-9 else float("nan"))

# ---- report ----------------------------------------------------------------------------------
print("\n%-6s %3s | %7s %7s | %6s %6s | %7s | %7s %8s %9s" %
      ("kind", "L", "EV", "EV(alt)", "L0", "cfg", "unseen", "dCE", "recov", "L_zero"))
for l in range(NL):
    for k in ("att", "mlp", "res"):
        r = rec[(k, l)]
        print("%-6s %3d | %7.3f %7.3f | %6.1f %6s | %6.1f%% | %+7.3f %8.3f %9.2f"
              % (k, l, r["ev"][V], r["ev"][1 - V], r["l0"][V], r["cfg_l0"], 100 * r["unseen"][V],
                 r["dCE"], r["recovered"], r["L_zero"]))
print("\nsummary by kind (median over 18 layers):")
for k in ("att", "mlp", "res"):
    rs = [rec[(k, l)] for l in range(NL)]
    print("  %-3s EV %.3f | L0 %.0f (cfg %s) | unseen %.1f%% | dCE %+.3f nats | recovered %.3f"
          % (k, np.median([r["ev"][V] for r in rs]), np.median([r["l0"][V] for r in rs]),
             sorted({r["cfg_l0"] for r in rs}), 100 * np.median([r["unseen"][V] for r in rs]),
             np.median([r["dCE"] for r in rs]), np.median([r["recovered"] for r in rs])))
json.dump(dict(model=MODEL, width=GS.WIDTH, l0=GS.L0, sub_bdec=bool(SUB), L_clean=L_clean, n_seq=N_SEQ, seq=SEQ,
               sites={"%s_%d" % s: r for s, r in rec.items()}),
          open(HERE / ("sae_recon_%s_%s.json" % (GS.WIDTH, GS.L0)), "w"), indent=1)
print("-> sae_recon_%s_%s.json" % (GS.WIDTH, GS.L0))
