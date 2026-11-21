"""129 F (CPU only): where do the circuits' RESIDUAL-site members get their value from?

Under WCM's circuit-only run every non-member latent at every site is ablated and member resid latents are re-encoded
from the edited stream, so a resid member cannot carry information whose writers were ablated. What a resid member
CAN carry without any head is what comes from the token + position embedding (not a site, never ablated). This splits
each resid / resid_mid member's natural pre-activation exactly over the stream's writers, in the unedited model:

  stream(L, kind) = embed + sum_{l < L} (attn_l + mlp_l) [+ attn_L (resid_mid) | + attn_L + mlp_L (resid)]
  x_n = (stream - mean) / scale  (the SAE's per-token layer norm; centring is linear, the scale is shared)
  pre_i = sum_c w_i . (c - mean c) / scale + b_eff_i

Per target and circuit, over held-out task prompts, for every member that is active (> 0, Top-K) at a token, the
share of sum_c |term_c| ... reported as the signed share of the non-bias pre-activation from each writer group
(embed, L0 attn, L0 mlp, attn L1+, mlp L1+), weighted by the member's natural activation, per token role.
(Check: the terms + b_eff reproduce the encoder pre-activation; max |diff| printed.)

  CUDA_VISIBLE_DEVICES= ./.venv/bin/python -I experiments/129-known-circuits-diagnosis/f_embed.py -> out/f_embed.json/.md
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E117 = ROOT / "experiments" / "117-known-circuits-gpt2"
sys.path.insert(0, str(E117))
import lib117 as L  # noqa: E402
import torch  # noqa: E402

N = int(os.environ.get("N_PROMPTS", "64"))
GROUPS = ("embed", "attn0", "mlp0", "attn1+", "mlp1+")


def components(inf, tok):
    """Writers of the residual stream: embed [B,T,d], attn[l], mlp[l] [B,T,d] (unedited forward)."""
    hf = inf.model.hf
    comp = {}
    hs = []
    for l, b in enumerate(hf.transformer.h):
        hs.append(b.attn.register_forward_hook(lambda m, i, o, l=l: comp.__setitem__(("attn", l), (o[0] if isinstance(o, tuple) else o).detach())))
        hs.append(b.mlp.register_forward_hook(lambda m, i, o, l=l: comp.__setitem__(("mlp", l), o.detach())))
    try:
        with torch.no_grad():
            hf(input_ids=tok, use_cache=False)
            pos = torch.arange(tok.shape[1])[None]
            comp["embed"] = (hf.transformer.wte(tok) + hf.transformer.wpe(pos)).detach()
    finally:
        for h in hs:
            h.remove()
    return comp


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "6")))
    from config import config
    from model.inference import Inference
    from sae.public_bank import PublicSAEBank
    config.model.family = "gpt2-small"
    inf = Inference(torch.device("cpu"), family="gpt2-small")
    bank = PublicSAEBank(device=torch.device("cpu"), family="gpt2-small", preload=False)
    eps = float(getattr(bank, "norm_eps", 1e-5))
    sel = json.loads((E117 / "out" / "targets.json").read_text())
    keys = [t for t in (E117 / "out" / "targets.txt").read_text().split() if t]
    tasks, res, maxdiff = {}, [], 0.0
    for key in keys:
        pname = next(p for p, d in sel["pathways"].items() if any(c["key"] == key for c in d["chosen"]))
        pw = L.PATHWAYS[pname]
        task = tasks.setdefault(pw["task"], L.load_task(pw["task"]))
        ent = next(c for c in sel["pathways"][pname]["chosen"] if c["key"] == key)
        items, toks = L.task_items(task, "held", pw["role"], ent.get("io_token"))
        items, toks = items[:N], toks[:N]
        roles = [L.roles_of(task["rows"][i]) for i in items]
        tok = L.pad_batch(toks, torch.device("cpu"))
        T = tok.shape[1]
        R = [r + ["pad"] * (T - len(r)) for r in roles]
        comp = components(inf, tok)
        for cname, sub in (("corpus", "main"), ("task", "task")):
            cp = E117 / "out" / "wcm" / sub / "circuits" / ("%s.pt" % key)
            if not cp.exists():
                continue
            c = torch.load(cp, weights_only=False, map_location="cpu")
            if c is None:
                continue
            mem = L.circuit_members(c)
            agg = defaultdict(lambda: defaultdict(float))     # role -> group -> weighted signed share
            wsum = defaultdict(float)
            for (layer, kind), d in mem.items():
                if kind not in ("resid", "resid_mid") or not d:
                    continue
                sae = bank.saes[kind][layer]
                idx = torch.tensor(sorted(d), dtype=torch.long)
                W = sae.encoder.weight.detach()[idx].float()                       # [m, d]
                b = sae._get_bias_eff().detach()[idx].float()
                parts = [("embed", comp["embed"])]
                for l in range(layer + 1):
                    parts.append(("attn0" if l == 0 else "attn1+", comp[("attn", l)]))
                    if l < layer or kind == "resid":
                        parts.append(("mlp0" if l == 0 else "mlp1+", comp[("mlp", l)]))
                stream = sum(p for _, p in parts)
                xc = stream - stream.mean(-1, keepdim=True)
                scale = xc.std(-1, keepdim=True) + eps
                pre = (xc / scale) @ W.T + b                                         # [B, T, m]
                ta, ti = bank.encode(xc / scale, kind, layer)
                # active = member in the natural Top-K set at that token
                act = torch.zeros_like(pre)
                for j, lat in enumerate(idx.tolist()):
                    m = (ti == lat) & (ta > 0)
                    act[..., j] = (ta * m).sum(-1)
                terms = defaultdict(lambda: torch.zeros_like(pre))
                for g, p in parts:
                    terms[g] = terms[g] + ((p - p.mean(-1, keepdim=True)) / scale) @ W.T
                tot = sum(terms[g] for g in GROUPS if g in terms)
                maxdiff = max(maxdiff, float((tot + b - pre).abs().max()))
                for bi in range(tok.shape[0]):
                    for t in range(len(roles[bi])):
                        a = act[bi, t]
                        on = a > 0
                        if not bool(on.any()):
                            continue
                        r = R[bi][t]
                        den = tot[bi, t][on]
                        for g in GROUPS:
                            if g in terms:
                                sh = terms[g][bi, t][on] / den.abs().clamp(min=1e-6)
                                agg[r][g] += float((sh * a[on]).sum())
                        wsum[r] += float(a[on].sum())
            out = {r: {g: agg[r][g] / wsum[r] for g in GROUPS} for r in agg if wsum[r] > 0}
            out = {r: dict(v, weight=wsum[r]) for r, v in out.items()}
            res.append(dict(seed=key, pathway=pname, circuit=cname, roles=out))
            print("[129 F] %s %s %s" % (key, cname, json.dumps({r: {g: round(v[g], 2) for g in GROUPS}
                                                                for r, v in out.items() if r in ("io", "s1", "s2", "end", "y1_yy")})), flush=True)
    (HERE / "out" / "f_embed.json").write_text(json.dumps(dict(rows=res, max_abs_diff=maxdiff), indent=1), encoding="utf-8")
    md = ["# 129 F: writer split of resid / resid_mid members' natural pre-activation (activation-weighted signed share "
          "of the non-bias term; held-out task prompts; check max |diff| %.2e)" % maxdiff, "",
          "| target | circuit | role | embed | attn L0 | mlp L0 | attn L1+ | mlp L1+ | weight |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        for ro, v in sorted(r["roles"].items()):
            if ro in ("pad",):
                continue
            md.append("| %s | %s | %s | %s | %.0f |" % (r["seed"], r["circuit"], ro,
                                                     " | ".join("%.2f" % v[g] for g in GROUPS), v["weight"]))
    (HERE / "out" / "f_embed.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("max diff", maxdiff)


if __name__ == "__main__":
    main()
