"""WHY DO ATTENTION-OUTPUT CIRCUITS FAIL? Test 3: per-head / per-source decomposition + upstream-kind ablation (2026-10-03).

Tests 1 (frozen patterns) and 2 (A/C weights, rank-keep) did not explain the failures; only circuit size helps. Two
hypotheses are tested here, both on the production circuits and the production contexts (held-out strongest):

  H1 (source positions). An attention latent at position t is a sum over heads h and SOURCE positions s of
     p[h,t,s] * (v[h,s] . u_h),  u_h = W_O_h^T w_seed. A circuit is one latent set kept at EVERY position, so content
     the target reads from other positions must survive in those members at those positions. If failing targets read
     more from other positions (not t itself, not BOS), and the A ablation removes mostly that part, the circuit has to
     carry many positions' content: a size problem, as observed.
     Under the A ablation the dominant head's loss is split into its QK part (ablated pattern, clean values) and its OV
     part (clean pattern, ablated values).
  H2 (upstream kind; Daniel's idea). Mean ablation might feed attention inputs it handles badly. The A ablation is
     restricted to ONE upstream kind (attn / mlp / resid sites only; members kept at fitted alpha, everything else
     at those sites at its positive-context mean, Top-K respected) and the target's retention r/a_pos is compared
     between attention targets and MLP / resid controls. resid sites are full-stream checkpoints in this hook layout
     (the block output), so "resid only" is close to the full ablation; attn / mlp sites are the per-layer increments.
  Also: the Top-K margin of the target at its anchor (clean), for the near-threshold censoring check.

Sample: every target of the freeze test (out_freeze: attention + MLP / resid controls, not near-threshold) plus
N_NEAR near-threshold attention targets per depth band. Pass labels come from targets.csv (production scores).

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/062-h100-protocol-v1/attn_heads.py
  env: N_NEAR (8)  KEYS (comma list, overrides the sample)  OUT_AH (default experiments/062-h100-protocol-v1/out_heads)  REPORT=1
"""
import json
import math
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
os.environ.setdefault("OUT", str(HERE / "out_full" / "out"))
sys.path.insert(0, str(HERE))
import driver  # noqa: E402

N_NEAR = int(os.environ.get("N_NEAR", 8))
OUT_AH = Path(os.environ.get("OUT_AH", str(HERE / "out_heads")))
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]

# capture state for the patched attention: at layer CAP["layer"], per batch row b, contributions at anchor CAP["anc"][b]
CAP = dict(layer=None, anc=None, u=None, out=None)


def band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def install(R):
    """Attention impl that, at the capture layer, records per row the anchor's pattern p[h, :] and the value
    projections vp[h, s] = v[h, s] . u_h (u_h = W_O_h^T w_seed); the output itself is computed exactly as before."""
    import model.turingllm as T
    inference = R.G["inference"]
    for i, block in enumerate(inference.model.transformer.h):
        block.attn._layer_idx = i
    orig = T.CausalSelfAttention._forward_impl

    def impl(self, x):
        if CAP["layer"] is None or getattr(self, "_layer_idx", None) != CAP["layer"]:
            return orig(self, x)
        B, Tn, C = x.size()
        nh = self.n_head; hd = C // nh
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (t.view(B, Tn, nh, hd).transpose(1, 2) for t in (q, k, v))
        s = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(hd)
        mask = torch.ones(Tn, Tn, dtype=torch.bool, device=x.device).tril()
        att = F.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
        y = (att.to(v.dtype) @ v).transpose(1, 2).contiguous().view(B, Tn, C)
        anc = CAP["anc"].to(x.device).clamp(0, Tn - 1)[:B]
        rr = torch.arange(B, device=x.device)
        u = CAP["u"].to(x.device)                                         # [nh, hd]
        vp = torch.einsum("bhsd,hd->bhs", v.float(), u)                   # [B, nh, T]
        CAP["out"] = dict(p=att[rr, :, anc, :].detach(), vp=vp.detach(), anc=anc.detach())   # p: [B, nh, T]
        return self.c_proj(y)
    T.CausalSelfAttention._forward_impl = impl
    inference.disable_compile()
    inference.enable_compile = lambda: None


def decompose(p, vp, anc):
    """Per-row metrics from p [B, nh, T] and vp [B, nh, T]; contributions c[b, h, s] = p * vp."""
    c = p * vp
    B, nh, Tn = c.shape
    out = []
    for b in range(B):
        t = int(anc[b])
        cb = c[b, :, :t + 1]                                              # causal: sources 0..t
        by_src = cb.sum(0); by_head = cb.sum(1)
        pos_src = by_src.clamp(min=0); pos_head = by_head.clamp(min=0)
        ms, mh = float(pos_src.sum()), float(pos_head.sum())
        self_c = float(by_src[t]); bos_c = float(by_src[0]) if t > 0 else 0.0
        dist = torch.arange(t, -1, -1, device=c.device, dtype=torch.float32)   # t - s
        out.append(dict(
            total=float(by_src.sum()), self_=self_c, bos=bos_c, other=float(by_src.sum()) - self_c - bos_c,
            frac_other_pos=(float(pos_src[1:t].sum()) / ms) if ms > 0 and t > 1 else 0.0,
            neff_src=(ms ** 2 / float((pos_src ** 2).sum())) if ms > 0 else 0.0,
            dist=(float((pos_src * dist).sum()) / ms) if ms > 0 else 0.0,
            top_head=int(by_head.argmax()), top_share=(float(pos_head.max()) / mh) if mh > 0 else 0.0,
            neff_head=(mh ** 2 / float((pos_head ** 2).sum())) if mh > 0 else 0.0,
            anc=t))
    return out


def setup_target(R, key, c):
    """The scorer's own preamble (amp_eval_pass_v2.score_circuit): contexts, split, anchors, members, gains."""
    V, G = R.V, R.G
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_anchors
    from pipeline.component_index import component_idx as comp_of
    bank, M0, KINDS, NK, D = (G[k] for k in ("bank", "M0", "KINDS", "NK", "D"))
    layer, kind, sl = key.split(".")[0], key.split(".")[1], key.split(".")[2]
    layer, sl = int(layer), int(sl)
    pd_ = M0.build_probe_dataset(comp_of(layer, KINDS.index(kind), NK), sl)
    pt, pa = pd_.pos_tokens[:V.N_SEQ], pd_.pos_argmax[:V.N_SEQ]
    n_tr = V.split_n(int(pt.shape[0]))
    seed = V.seed_of(c)
    alphas = defaultdict(dict)
    for n in c.nodes.values():
        if n is seed:
            continue
        f = n.metadata["feature_id"]; alphas[(f.layer, f.kind)][int(f.index)] = float(n.metadata.get("amplitude", 1.0))
    dev = pt.device
    msets = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in alphas.items() if d}
    scales = {}
    for s, v in msets.items():
        sv = torch.ones(D, device=dev, dtype=torch.float32)
        sv[v] = torch.tensor([alphas[s][int(i)] for i in v.tolist()], device=dev, dtype=torch.float32); scales[s] = sv
    UPS = set(upstream_sites(bank, layer, kind))
    means_tr, _ = collect_site_anchors(G["inference"], bank, pt[:n_tr], UPS, pa[:n_tr], pin_position_specific=False)
    sae = bank.saes[kind][layer]
    return dict(layer=layer, kind=kind, sl=sl, pt=pt[n_tr:], pa=pa[n_tr:], alphas=dict(alphas), msets=msets,
                scales=scales, UPS=UPS, means=means_tr, sae=sae,
                w=sae.encoder.weight[sl].detach(), b=sae._get_bias_eff()[sl].detach())


def run_target(R, key, c):
    V, G = R.V, R.G
    bank, K, D = G["bank"], int(G["K"]), G["D"]
    TapCO, _ = V._engine_classes()
    S = setup_target(R, key, c)
    layer, kind, sl, pt, pa = S["layer"], S["kind"], S["sl"], S["pt"], S["pa"]
    inference = G["inference"]
    row = dict(seed=key, layer=layer, kind=kind, n=sum(len(d) for d in S["alphas"].values()), n_ho=int(pt.shape[0]))
    # member composition: count by kind, and the share sitting at the layer directly below the target
    comp = defaultdict(int)
    for (l_, k_), d in S["alphas"].items():
        comp[k_] += len(d)
        if l_ == layer - 1 or (l_ == layer and k_ != kind):
            comp["adjacent"] += len(d)
    row.update({"mem_" + k: v for k, v in comp.items()})

    attn_target = kind == "attn"
    if attn_target:
        blk = inference.model.transformer.h[layer].attn
        nh = blk.n_head; hd = blk.n_embd // nh
        Wo = blk.c_proj.weight.detach().float()                           # [C, C]: y = z Wo^T + bo
        CAP["u"] = torch.stack([Wo[:, h * hd:(h + 1) * hd].T @ S["w"].float() for h in range(nh)])   # [nh, hd]
        row["bias_part"] = float(blk.c_proj.bias.detach().float() @ S["w"].float() + S["b"].float())

    def fwd(patcher, cap):
        CAP.update(layer=layer if (cap and attn_target) else None, anc=pa, out=None)
        with torch.no_grad():
            inference.forward(pt, patcher=patcher, grad_enabled=False, return_activations=False, tokenize_final=False)
        CAP["layer"] = None
        rr = torch.arange(int(pt.shape[0]), device=patcher.seed_pre.device)
        anc = pa.to(patcher.seed_pre.device).clamp(0, patcher.seed_pre.shape[1] - 1)
        return patcher.seed_pre[rr, anc].float(), patcher.tap_tk[rr, anc].float(), CAP["out"]

    # ---- clean run (+ Top-K margin at the anchor: seed pre-activation vs the largest pre-activation left out of Top-K)
    class Clean(V.AmpInjectPatcher):
        def transform(self, layer_idx, kind_, x):
            if (layer_idx, kind_) == self.seed_site:
                rr = torch.arange(x.shape[0], device=x.device); anc = pa.to(x.device).clamp(0, x.shape[1] - 1)
                xa = x[rr, anc].float()
                pre_all = xa @ S["sae"].encoder.weight.detach().float().T + S["sae"]._get_bias_eff().detach().float()
                top = pre_all.topk(K + 1, dim=-1).values
                self.margin = ((pre_all[:, sl] - top[:, K]) / pre_all[:, sl].clamp(min=1e-6)).cpu()
                self.in_topk = (pre_all[:, sl] >= top[:, K - 1]).cpu()
            return super().transform(layer_idx, kind_, x)
    pc = Clean({}, (layer, kind), S["w"], S["b"], sl)
    pre0, tk0, cap0 = fwd(pc, True)
    a_pos_tk, a_pos_pre = float(tk0.mean()), float(torch.relu(pre0).mean())
    row.update(a_pos_tk=a_pos_tk, a_pos_pre=a_pos_pre, margin_med=float(pc.margin.median()),
               margin_lt5=float((pc.margin < 0.05).float().mean()), in_topk=float(pc.in_topk.float().mean()))

    def co(scope, cap=False):
        p = TapCO(bank=bank, keep_indices={s: set(d) for s, d in S["alphas"].items() if d}, in_scope=scope,
                  seed_layer=layer, seed_kind=kind, seed_latent_idx=sl, pos_argmax=pa, site_means=S["means"],
                  respect_topk=True, topk=K, keep_tensors=dict(S["msets"]), keep_scales=S["scales"] or None,
                  w_seed=S["w"], b_seed=S["b"])
        return fwd(p, cap)

    # ---- H2: A ablation restricted to one upstream kind; retention r / a_pos under both reads
    scopes = {"all": S["UPS"]}
    for k_ in G["KINDS"]:
        scopes[k_] = {s for s in S["UPS"] if s[1] == k_}
    capA = None
    for name, scope in scopes.items():
        if not scope:
            continue
        pre_, tk_, cap_ = co(scope, cap=(name == "all"))
        row["ret_tk_" + name] = float(tk_.mean()) / a_pos_tk if a_pos_tk > 0 else None
        row["ret_pre_" + name] = float(torch.relu(pre_).mean()) / a_pos_pre if a_pos_pre > 0 else None
        if name == "all":
            capA = cap_

    # ---- H1: decomposition, clean vs A ablation (all upstream sites), and the dominant head's QK / OV split
    if attn_target and cap0 is not None and capA is not None:
        recon = (cap0["p"] * cap0["vp"]).sum((1, 2)) + row["bias_part"]
        row["recon_err"] = float((recon - pre0.to(recon.device)).abs().max())
        m0, mA = decompose(cap0["p"], cap0["vp"], cap0["anc"]), decompose(capA["p"], capA["vp"], capA["anc"])
        for f in ("total", "self_", "bos", "other", "frac_other_pos", "neff_src", "dist", "top_share", "neff_head"):
            row["c_" + f] = sum(r[f] for r in m0) / len(m0)
            row["A_" + f] = sum(r[f] for r in mA) / len(mA)
        # dominant head: the head with the largest mean positive contribution on the clean run
        hc = (cap0["p"] * cap0["vp"]).sum(-1).mean(0)
        h = int(hc.argmax())
        row["dom_head"] = h; row["dom_share"] = float(hc.clamp(min=0)[h] / hc.clamp(min=0).sum().clamp(min=1e-9))
        p0, v0, pA, vA = cap0["p"][:, h], cap0["vp"][:, h], capA["p"][:, h], capA["vp"][:, h]
        base = float((p0 * v0).sum(-1).mean())
        row["dom_clean"] = base
        row["dom_A"] = float((pA * vA).sum(-1).mean())
        row["dom_qk_only"] = float((pA * v0).sum(-1).mean())                # ablated pattern, clean values
        row["dom_ov_only"] = float((p0 * vA).sum(-1).mean())                # clean pattern, ablated values
        # attention entropy of the dominant head at the anchor, clean and ablated
        ent = lambda p: float((-(p.clamp(min=1e-12)).log() * p).sum(-1).mean())
        row["dom_ent_clean"], row["dom_ent_A"] = ent(p0), ent(pA)
    return row


def sample():
    import pandas as pd
    if os.environ.get("KEYS"):
        return os.environ["KEYS"].split(",")
    keys = []
    fz =HERE / "out_freeze" / "freeze.jsonl"
    for line in open(fz):
        k = json.loads(line)["seed"]
        if k not in keys:
            keys.append(k)
    tg = pd.read_csv(driver.OUT / "targets.csv", index_col="seed")
    have = {p.stem for p in (driver.OUT / "main" / "circuits").glob("*.pt")}
    tg = tg[tg.index.isin(have)]
    rng = random.Random(20261003)
    for b in ("L0-4", "L5-7", "L8-11"):
        pool = sorted(k for k in tg[(tg.kind == "attn") & (tg.rank_clean >= 64)].index if band(int(k.split(".")[0])) == b)
        keys += rng.sample(pool, min(N_NEAR, len(pool)))
    return keys


def main():
    OUT_AH.mkdir(parents=True, exist_ok=True)
    path = OUT_AH / "heads.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add(r["seed"])
            except Exception:  # noqa: BLE001
                pass
    keys = sample()
    print("heads test: %d targets (%d done)" % (len(keys), len(done)), flush=True)
    R = driver.Runner()
    install(R)
    fh = open(path, "a")
    t0 = time.time()
    for n, key in enumerate(keys):
        if key in done:
            continue
        c = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
        if c is None:
            continue
        try:
            R.H.patch_eval_contexts(R.G, R.contexts(key), "strong")
            row = run_target(R, key, c)
        except Exception as e:  # noqa: BLE001
            row = dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        CAP.update(layer=None, out=None)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        el = time.time() - t0
        print("%3d/%d %-16s %s %.0fs elapsed" % (n + 1, len(keys), key,
                                                "ERR" if "error" in row else "ok", el), flush=True)
    print("DONE", flush=True)


def report():
    import pandas as pd
    d = pd.DataFrame([json.loads(l) for l in open(OUT_AH / "heads.jsonl")])
    d = d[d.get("error", pd.Series(index=d.index, dtype=object)).isna()].drop_duplicates("seed", keep="last")
    tg = pd.read_csv(driver.OUT / "targets.csv", index_col="seed")
    t = tg.loc[d.seed]
    d["pass"] = (((t[HEAD] >= 0.8) & (t[HEAD] <= 1.5)).all(axis=1) & (t.phi_sup_blind_tk >= 0.9)).values
    d["near"] = (t.rank_clean >= 64).values
    d["band"] = d.layer.map(band)
    d["grp"] = d.kind + d["pass"].map({True: " pass", False: " fail"}) + d.near.map({True: " near", False: ""})
    print("rows %d; max reconstruction error of the decomposition %.2e" % (len(d), d.recon_err.max()))
    pd.set_option("display.width", 200)
    print("\nTop-K margin at the anchor (clean): (seed - first excluded) / seed")
    print(d.groupby("grp").agg(k=("seed", "size"), margin=("margin_med", "median"), lt5=("margin_lt5", "median"),
                               in_topk=("in_topk", "median")).round(3).to_string())
    print("\nH2: retention r/a_pos (activation read) when the A ablation hits ONE upstream kind")
    cols = [c for c in ("ret_tk_attn", "ret_tk_mlp", "ret_tk_resid", "ret_tk_all") if c in d]
    print(d[~d.near].groupby(["kind", "pass"])[cols].median().round(2).to_string())
    print(d[~d.near].groupby(["kind", "band"])[cols].median().round(2).to_string())
    print("\nmember composition (share of members by kind)")
    for k_ in ("attn", "mlp", "resid", "adjacent"):
        d["sh_" + k_] = (d["mem_" + k_].fillna(0) if "mem_" + k_ in d else 0) / d.n
    print(d[~d.near].groupby(["kind", "pass"])[["n", "sh_attn", "sh_mlp", "sh_resid", "sh_adjacent"]].median().round(2).to_string())
    a = d[(d.kind == "attn") & ~d.near].copy()
    print("\nH1: attention targets, clean decomposition (means over held-out sequences; medians over targets)")
    f = ["c_frac_other_pos", "c_neff_src", "c_dist", "c_top_share", "c_neff_head", "dom_share"]
    print(a.groupby("pass")[f].median().round(2).to_string())
    print(a.groupby(["band", "pass"])[f].median().round(2).to_string())
    print("\nH1: what the A ablation removes (retention of each part = ablated / clean, medians)")
    for part in ("total", "self_", "bos", "other"):
        a["ret_" + part] = a["A_" + part] / a["c_" + part].where(a["c_" + part].abs() > 1e-6)
    a["ret_dom"] = a.dom_A / a.dom_clean
    a["ret_dom_qk"] = a.dom_qk_only / a.dom_clean
    a["ret_dom_ov"] = a.dom_ov_only / a.dom_clean
    print(a.groupby("pass")[["ret_total", "ret_self_", "ret_other", "ret_dom", "ret_dom_qk", "ret_dom_ov",
                             "dom_ent_clean", "dom_ent_A"]].median().round(2).to_string())
    print(a.groupby(["band", "pass"])[["ret_self_", "ret_other", "ret_dom_qk", "ret_dom_ov"]].median().round(2).to_string())
    print("\nshare of clean contribution from self / BOS / other (signed means, medians over targets)")
    for part in ("self_", "bos", "other"):
        a["sh_" + part] = a["c_" + part] / a.c_total.where(a.c_total.abs() > 1e-6)
    print(a.groupby("pass")[["sh_self_", "sh_bos", "sh_other"]].median().round(2).to_string())


if __name__ == "__main__":
    report() if os.environ.get("REPORT") else main()
