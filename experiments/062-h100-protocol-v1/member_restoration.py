"""DO THE COEFFICIENTS RESTORE MEMBERS TO THEIR NATURAL VALUES? (Daniel's idea, 2026-10-06)

In a circuit-only run every non-member is ablated, so each member is re-encoded from an ablated stream and then scaled
by its fitted coefficient alpha (CircuitOnlyPatcher: kept value = alpha x the member's post-Top-K encoding at its
site). This script compares, per member, at the target's anchor positions on the held-out strongest contexts:
  nat        the member's activation in the unmodified model (no ablation)
  abl[F]     its activation inside the circuit-only run under fill F (Z zero, A activating mean, C contrast mean),
             BEFORE the coefficient
  amp[F]     alpha x abl[F], the value the circuit actually writes forward
If amp ~ nat, the coefficients put members back where they naturally sit (restoration); if amp >> nat, the circuit
drives members above their natural level (amplification); members with nat = 0 and amp > 0 are switched on.
Also the mean over ALL positions (members feed attention at every position, not only at the anchor).

Follow-up checks (v2, 2026-10-06):
  attr       each member's gradient x activation contribution to the target's pre-activation in the unmodified run
             (summed over positions, read at the anchors; the scorer's own role attribution), to weight members
  target_a1_[F]  the circuit-only target with every alpha set to 1 (members at their unscaled ablated values): the
             share of the target the members carry alone under that fill; 1 - share = what the ablated non-members
             supplied. Tested against the mass the coefficients add (sum alpha x abl / sum abl).
  all-position means are summarised next to the anchor reads.

Sample: full-run circuits, N_PER per depth band, fixed seed, from the production circuits (main/circuits). Resumable.

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/062-h100-protocol-v1/member_restoration.py
  env: N_PER (60)  OUT_MR (default experiments/062-h100-protocol-v1/out_restore)  REPORT=1
"""
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
os.environ.setdefault("OUT", str(HERE / "out_full" / "out"))
sys.path.insert(0, str(HERE))
import driver  # noqa: E402
import attn_heads as AH  # noqa: E402  (setup_target: the scorer's own preamble)

N_PER = int(os.environ.get("N_PER", 60))
OUT_MR = Path(os.environ.get("OUT_MR", str(HERE / "out_restore")))
RFILE = os.environ.get("RFILE", "restore_v2.jsonl")         # v1 (restore.jsonl, 2026-10-06) had no attr / alpha = 1
FILLS = ("Z", "A", "C")


def band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def sample():
    import pandas as pd
    tg = pd.read_csv(driver.OUT / "targets.csv", index_col="seed")
    have = {p.stem for p in (driver.OUT / "main" / "circuits").glob("*.pt")}
    keys = sorted(k for k in tg.index if k in have)
    rng = random.Random(20261006)
    out = []
    for b in ("L0-4", "L5-7", "L8-11"):
        pool = [k for k in keys if band(int(k.split(".")[0])) == b]
        out += rng.sample(pool, min(N_PER, len(pool)))
    rng.shuffle(out)                                     # a partial run still covers every band
    return out


def run_target(R, key, c):
    from eval.floors import collect_site_means
    from sae.dense import sparse_topk_to_dense
    V, G = R.V, R.G
    bank, K, D = G["bank"], int(G["K"]), G["D"]
    TapCO, _ = V._engine_classes()
    rec = R.contexts(key)
    R.H.patch_eval_contexts(G, rec, "strong")
    S = AH.setup_target(R, key, c)
    pt, pa = S["pt"], S["pa"]
    nt = rec["neg"].to(G["device"])
    means_neg = collect_site_means(G["inference"], bank, nt[:V.split_n(int(nt.shape[0]))], S["UPS"])
    sites = sorted(S["msets"])
    idx = {s: S["msets"][s] for s in sites}

    def capture(store, layer_idx, kind, x):
        s = (layer_idx, kind)
        if s in idx:
            ta, ti = bank.encode(x, kind, layer_idx)
            dense = sparse_topk_to_dense(ta, ti, D, dtype=torch.float32)[..., idx[s]]       # [B, T, m]
            rr = torch.arange(x.shape[0], device=x.device)
            anc = pa.to(x.device).clamp(0, x.shape[1] - 1)[:x.shape[0]]
            store[s] = (dense[rr, anc].mean(0).cpu(), dense.mean((0, 1)).cpu())

    class Clean(V.AmpInjectPatcher):
        def transform(self, layer_idx, kind, x):
            capture(self.cap, layer_idx, kind, x)
            return super().transform(layer_idx, kind, x)

    class Cap(TapCO):
        def transform(self, layer_idx, kind, x):
            capture(self.cap, layer_idx, kind, x)              # the live encoding, BEFORE alpha and the fill
            return super().transform(layer_idx, kind, x)

    inf = G["inference"]

    def fwd(p):
        p.cap = {}
        inf.disable_compile()
        try:
            with torch.no_grad():
                inf.forward(pt, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
        finally:
            inf.enable_compile()
        rr = torch.arange(int(pt.shape[0]), device=p.tap_tk.device)
        anc = pa.to(p.tap_tk.device).clamp(0, p.tap_tk.shape[1] - 1)
        tgt = float(p.tap_tk[rr, anc].mean())
        p.cap["_pre"] = float(torch.relu(p.seed_pre[rr, anc.to(p.seed_pre.device)].float()).mean())   # relu(w.x + b)
        return p.cap, tgt

    clean, t_nat = fwd(Clean({}, (S["layer"], S["kind"]), S["w"], S["b"], S["sl"]))
    fills = {"Z": (None, False), "A": (S["means"], True), "C": (means_neg, True)}
    keep = {s: set(d) for s, d in S["alphas"].items() if d}
    row = dict(seed=key, layer=S["layer"], kind=S["kind"], n=sum(len(v) for v in idx.values()), target_nat=t_nat,
               target_pre_nat=clean["_pre"])
    alpha = np.concatenate([np.array([S["alphas"][s][int(i)] for i in idx[s].tolist()]) for s in sites])
    mlayer = np.concatenate([np.full(len(idx[s]), s[0]) for s in sites])
    mkind = sum([[s[1]] * len(idx[s]) for s in sites], [])
    row.update(alpha=alpha.round(4).tolist(), mlayer=mlayer.tolist(), mkind=mkind,
               nat=np.concatenate([clean[s][0].numpy() for s in sites]).round(4).tolist(),
               nat_all=np.concatenate([clean[s][1].numpy() for s in sites]).round(4).tolist())
    for f, (mm, topk) in fills.items():
        for scaled in (True, False):
            p = Cap(bank=bank, keep_indices=keep, in_scope=S["UPS"], seed_layer=S["layer"], seed_kind=S["kind"],
                    seed_latent_idx=S["sl"], pos_argmax=pa, site_means=mm, respect_topk=topk, topk=K,
                    keep_tensors=dict(S["msets"]), keep_scales=S["scales"] if scaled else None,
                    w_seed=S["w"], b_seed=S["b"])
            cap, t = fwd(p)
            if scaled:
                row["abl_" + f] = np.concatenate([cap[s][0].numpy() for s in sites]).round(4).tolist()
                row["abl_all_" + f] = np.concatenate([cap[s][1].numpy() for s in sites]).round(4).tolist()
                row["target_" + f] = t
                row["target_pre_" + f] = cap["_pre"]
            else:                                                     # every alpha = 1
                row["target_a1_" + f] = t
                row["target_pre_a1_" + f] = cap["_pre"]
                row["abl_a1_" + f] = np.concatenate([cap[s][0].numpy() for s in sites]).round(4).tolist()

    # member contributions in the unmodified run: grad x activation of the target's anchor pre-activation
    msets = {s: idx[s] for s in sites}
    attr = {s: torch.zeros(len(v), device=pt.device) for s, v in msets.items()}
    wdec = {s: bank.saes[s[1]][s[0]].decoder.weight.detach()[:, v.to(bank.saes[s[1]][s[0]].decoder.weight.device)]
            .to(device=pt.device, dtype=torch.float32) for s, v in msets.items()}
    inf.disable_compile()
    try:
        for s0 in range(0, int(pt.shape[0]), V.ROLE_BS):
            tkb, anb = pt[s0:s0 + V.ROLE_BS], pa[s0:s0 + V.ROLE_BS]
            rp = V.RolePatcher(msets, (S["layer"], S["kind"]), S["w"], S["b"])
            inf.forward(tkb, patcher=rp, grad_enabled=True, return_activations=False, tokenize_final=False)
            pre = rp.seed_pre
            anc = anb.to(pre.device).clamp(0, pre.shape[1] - 1)
            tgt = pre[torch.arange(pre.shape[0], device=pre.device), anc].float().sum()
            ss = list(rp.z)
            gs = torch.autograd.grad(tgt, [rp.z[s] for s in ss], allow_unused=True)
            for s, g in zip(ss, gs):
                if g is not None:
                    attr[s] += ((g.detach().float() @ wdec[s]) * rp.codes[s]).sum(dim=(0, 1))
            del rp, gs, tgt, pre
    finally:
        inf.enable_compile()
    row["attr"] = np.concatenate([(attr[s] / int(pt.shape[0])).cpu().numpy() for s in sites]).round(5).tolist()
    return row


def main():
    OUT_MR.mkdir(parents=True, exist_ok=True)
    path = OUT_MR / RFILE
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
    print("member restoration: %d circuits (%d done)" % (len(keys), len(done)), flush=True)
    R = driver.Runner()
    fh = open(path, "a")
    t0 = time.time()
    for n, key in enumerate(keys):
        if key in done:
            continue
        c = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
        if c is None:
            continue
        try:
            row = run_target(R, key, c)
        except Exception as e:  # noqa: BLE001
            row = dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("%3d/%d %-16s %s %.0fs" % (n + 1, len(keys), key, "ERR" if "error" in row else "ok n=%d" % row["n"],
                                         time.time() - t0), flush=True)
    print("DONE", flush=True)


def report_v2():
    """The follow-up checks: contribution-weighted restoration, the share the members carry at alpha = 1 against the
    contribution the coefficients add, and the all-position reads."""
    import pandas as pd
    rows = [json.loads(l) for l in open(OUT_MR / RFILE)]
    rows = [r for r in rows if "error" not in r]
    tg = pd.read_csv(driver.OUT / "targets.csv", index_col="seed")
    HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]
    recs = []
    for r in rows:
        t = tg.loc[r["seed"]]
        passes = bool(((t[HEAD] >= 0.8) & (t[HEAD] <= 1.5)).all() and t.phi_sup_blind_tk >= 0.9)
        nat, a, at = np.array(r["nat"]), np.array(r["alpha"]), np.array(r["attr"])
        nat_all = np.array(r["nat_all"])
        for f in FILLS:
            abl, abl_all = np.array(r["abl_" + f]), np.array(r["abl_all_" + f])
            on = nat > 1e-6
            rr = np.zeros_like(nat); rr[on] = abl[on] / nat[on]           # ablated / natural, before alpha
            post = a * rr
            w = np.abs(at) * on
            W = w.sum()
            pos = at > 0
            # contribution-weighted effect on the target, linear in each member's activation: what share of the
            # members' natural contribution reaches the target in the circuit run, before and after alpha
            S_nat = at[on].sum()
            carry_a1 = float((at[on] * rr[on]).sum() / S_nat) if abs(S_nat) > 1e-9 else np.nan
            carry_amp = float((at[on] * post[on]).sum() / S_nat) if abs(S_nat) > 1e-9 else np.nan
            on_all = nat_all > 1e-6
            post_all = a[on_all] * abl_all[on_all] / nat_all[on_all]
            tn = r["target_nat"]
            recs.append(dict(
                seed=r["seed"], band=band(r["layer"]), kind=r["kind"], passes=passes, fill=f,
                w_restored=float((w * ((post >= 0.8) & (post <= 1.25))).sum() / W) if W > 0 else np.nan,
                w_above=float((w * (post > 1.25)).sum() / W) if W > 0 else np.nan,
                w_below=float((w * (post < 0.8)).sum() / W) if W > 0 else np.nan,
                w_dead=float((w * (abl == 0)).sum() / W) if W > 0 else np.nan,
                top10_share=float(np.sort(w)[::-1][:max(1, int(0.1 * on.sum()))].sum() / W) if W > 0 else np.nan,
                share_a1=r["target_a1_" + f] / tn if tn > 0 else np.nan,      # target, members alone at alpha = 1
                share_fit=r["target_" + f] / tn if tn > 0 else np.nan,        # target, fitted circuit
                # the same on the pre-activation read, relu(w.x + b): not censored by the site's Top-K
                pshare_a1=r["target_pre_a1_" + f] / r["target_pre_nat"] if r["target_pre_nat"] > 0 else np.nan,
                pshare_fit=r["target_pre_" + f] / r["target_pre_nat"] if r["target_pre_nat"] > 0 else np.nan,
                carry_a1=carry_a1, carry_amp=carry_amp,
                gain=carry_amp / carry_a1 if carry_a1 and carry_a1 > 0 else np.nan,
                alpha_attr_r=float(pd.Series(a[w > 0]).corr(pd.Series(w[w > 0]), method="spearman"))
                if (w > 0).sum() > 10 else np.nan,
                alpha_pos=float(np.median(a[pos & on])) if (pos & on).any() else np.nan,
                alpha_neg=float(np.median(a[~pos & on])) if (~pos & on).any() else np.nan,
                post_med_all=float(np.median(post_all)) if len(post_all) else np.nan,
                restored_all=float(((post_all >= 0.8) & (post_all <= 1.25)).mean()) if len(post_all) else np.nan))
    d = pd.DataFrame(recs)
    pd.set_option("display.width", 230)
    cols = ["w_restored", "w_above", "w_below", "w_dead", "top10_share", "share_a1", "share_fit", "pshare_a1",
            "pshare_fit", "carry_a1", "carry_amp", "gain", "alpha_attr_r", "alpha_pos", "alpha_neg", "post_med_all",
            "restored_all"]
    print("circuits %d" % d.seed.nunique())
    print("\nmedians over circuits, by ablation method:")
    print(d.groupby("fill")[cols].median().round(2).to_string())
    print("\nby ablation method x depth band:")
    print(d.groupby(["fill", "band"])[cols].median().round(2).to_string())
    print("\nby ablation method x pass:")
    print(d.groupby(["fill", "passes"])[cols].median().round(2).to_string())
    print("\ndoes the coefficients' gain match the members' shortfall at alpha = 1? (Spearman, log gain vs "
          "log 1/share_a1, circuits with both > 0)")
    for f in FILLS:
        for col in ("share_a1", "pshare_a1"):
            g = d[(d.fill == f) & (d.gain > 0) & (d[col] > 0)]
            if len(g) > 5:
                print("  %s %-9s: r = %.2f over %d circuits; median gain %.2f, median 1/share %.2f" % (
                    f, col, np.log(g.gain).corr(np.log(1 / g[col]), method="spearman"), len(g), g.gain.median(),
                    (1 / g[col]).median()))
    return d


def figure():
    """APPENDIX FIGURE (2026-10-06, approved by Daniel; fig:coef-members), drawn at the paper's text width.
    (a) per member (pooled over the sampled circuits, members active at the anchor in the unmodified run), its value
        in the circuit-only run over its natural value, under activating-mean ablation: before the coefficient
        (grey) and after it (blue); the +-25% band shaded; members pushed out of their site's Top-K read 0.
    (b) the target in the circuit-only run as a share of its natural activation (median over circuits), with every
        alpha = 1 (members alone at their unscaled values, grey) and with the fitted coefficients (blue), per
        ablation method and target depth band."""
    import pandas as pd
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from analysis.style import (BLUE, INK, INK_SECONDARY, SURFACE, configure_matplotlib, save_figure,  # noqa
                                tint)
    rows = [json.loads(l) for l in open(OUT_MR / RFILE)]
    rows = [r for r in rows if "error" not in r]
    pre, post = [], []
    for r in rows:
        nat, a, abl = np.array(r["nat"]), np.array(r["alpha"]), np.array(r["abl_A"])
        on = nat > 1e-6
        pre.append(abl[on] / nat[on]); post.append(a[on] * abl[on] / nat[on])
    pre, post = np.concatenate(pre), np.concatenate(post)
    d = report_v2()
    plt = configure_matplotlib()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.6, "axes.titlepad": 4.0, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "legend.fontsize": 6.3, "axes.linewidth": 0.7,
                         "grid.linewidth": 0.5, "axes.labelpad": 2.0})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.3), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.62})
    # the main histogram covers 0 < ratio < 3; the two edge groups (exactly 0 = out of Top-K, and >= 3) are drawn as
    # separate side-by-side bar pairs so that both lines' edge masses are visible (Daniel, 2026-10-06)
    bins = np.linspace(0, 3, 61)
    ax.axvspan(0.8, 1.25, color=tint("#12c46a", 0.88), zorder=0, linewidth=0)
    series = ((pre, "#9aa0a8", "before $\\alpha$"), (post, BLUE, "after $\\alpha$"))
    for vals, col, lab in series:
        inner = vals[(vals > 0) & (vals < 3)]
        ax.hist(inner, bins=bins, weights=np.full(len(inner), 100.0 / len(vals)), histtype="step", color=col,
                linewidth=1.2, label=lab, zorder=3)
    top = 0
    for j, (vals, col, lab) in enumerate(series):
        for xc, share in ((-0.55, 100 * (vals == 0).mean()), (3.55, 100 * (vals >= 3).mean())):
            x = xc + (j - 0.5) * 0.24
            ax.bar(x, share, 0.22, color=col, zorder=3, linewidth=0)
            ax.text(x, share + 0.3, "%.0f" % share, ha="center", va="bottom", fontsize=5.6, color=col)
            top = max(top, share)
    for xs in (-0.18, 3.18):
        ax.axvline(xs, color=INK_SECONDARY, linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=2)
    for vals, col, yy, lab in ((pre, "#9aa0a8", 0.56, "before $\\alpha$"), (post, BLUE, 0.47, "after $\\alpha$")):
        ax.text(0.45, yy, "%s: %.0f%%\nwithin ±25%%" % (lab, 100 * ((vals >= 0.8) & (vals <= 1.25)).mean()),
                transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color=col, linespacing=0.95)
    ax.set_xticks([-0.55, 0.5, 1, 1.5, 2, 2.5, 3.55])
    ax.set_xticklabels(["0\n(off)", "0.5", "1", "1.5", "2", "2.5", "≥3"])
    ax.set(xlim=(-0.85, 3.85), ylim=(0, max(top, 16) * 1.12), xlabel="member value / natural value (anchor)",
           ylabel="members (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) Members under ablation A")

    fills, bands = ("Z", "A", "C"), ("L0-4", "L5-7", "L8-11")
    ylab = []
    y = 0
    for f in fills:
        for b in bands:
            g = d[(d.fill == f) & (d.band == b)]
            a1, fit = 100 * g.share_a1.median(), 100 * g.share_fit.median()
            bx.plot([a1, fit], [y, y], color=tint(INK_SECONDARY, 0.6), linewidth=1.0, zorder=1)
            bx.plot(a1, y, "o", color="#9aa0a8", markersize=4, zorder=3)
            bx.plot(fit, y, "o", color=BLUE, markersize=4, zorder=3)
            ylab.append("%s  %s" % (f, "layers " + b[1:].replace("-", "–")))
            y += 1
        y += 0.6
    pos = [i + 0.6 * (i // 3) for i in range(9)]
    bx.set_yticks(pos); bx.set_yticklabels(ylab); bx.invert_yaxis()
    bx.axvline(100, color=INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)))
    bx.set(xlim=(-3, 112), xlabel="target, % of its natural activation (median)")
    bx.grid(axis="y", visible=False); bx.grid(axis="x", visible=True)
    bx.set_title("(b) The target: all $\\alpha = 1$ (grey) vs fitted (blue)")
    png = save_figure(fig, OUT_MR / "coefficients-members.png")
    import shutil
    shutil.copy(png.with_suffix(".pdf"), HERE.parents[1] / "paper" / "figures" / "coefficients-members.pdf")
    print("figure:", png, "-> paper/figures/coefficients-members.pdf (fig:coef-members, app:weighted-details)")
    print("pooled members (A): %d | before alpha in +-25%%: %.1f%%, after: %.1f%% | out of Top-K: %.1f%% | "
          "after alpha median %.2f, above 1.25: %.1f%%, below 0.8: %.1f%%" % (
              len(pre), 100 * ((pre >= 0.8) & (pre <= 1.25)).mean(), 100 * ((post >= 0.8) & (post <= 1.25)).mean(),
              100 * (pre == 0).mean(), np.median(post), 100 * (post > 1.25).mean(), 100 * (post < 0.8).mean()))


def report():
    if os.environ.get("FIG"):
        return figure()
    if RFILE != "restore.jsonl":
        return report_v2()
    import pandas as pd
    rows = [json.loads(l) for l in open(OUT_MR / "restore.jsonl")]
    rows = [r for r in rows if "error" not in r]
    tg = pd.read_csv(driver.OUT / "targets.csv", index_col="seed")
    HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]
    recs = []
    for r in rows:
        t = tg.loc[r["seed"]]
        passes = bool(((t[HEAD] >= 0.8) & (t[HEAD] <= 1.5)).all() and t.phi_sup_blind_tk >= 0.9)
        nat, a = np.array(r["nat"]), np.array(r["alpha"])
        for f in FILLS:
            abl = np.array(r["abl_" + f])
            amp = a * abl
            on = nat > 1e-6
            rat = amp[on] / nat[on]
            pre = abl[on] / nat[on]
            recs.append(dict(
                seed=r["seed"], band=band(r["layer"]), kind=r["kind"], passes=passes, fill=f, n=len(nat),
                active=on.mean(),
                pre_med=float(np.median(pre)) if on.any() else np.nan,      # ablated / natural, before alpha
                post_med=float(np.median(rat)) if on.any() else np.nan,     # alpha x ablated / natural
                restored=float(((rat >= 0.8) & (rat <= 1.25)).mean()) if on.any() else np.nan,
                above=float((rat > 1.25).mean()) if on.any() else np.nan,
                below=float((rat < 0.8).mean()) if on.any() else np.nan,
                dead=float((abl[on] == 0).mean()) if on.any() else np.nan,  # active naturally, 0 in the ablated run
                switched_on=float((amp[~on] > 0).mean()) if (~on).any() else np.nan,
                mass_pre=float(abl.sum() / max(nat.sum(), 1e-9)),          # summed activation, ablated / natural
                mass_post=float(amp.sum() / max(nat.sum(), 1e-9)),
                alpha_med=float(np.median(a)),
                # does alpha compensate for the drop? correlation of log alpha with log(nat / abl)
                comp_r=float(pd.Series(np.log(a[on & (abl > 0)])).corr(
                    pd.Series(np.log(nat[on & (abl > 0)] / abl[on & (abl > 0)])), method="spearman"))
                if (on & (abl > 0)).sum() > 10 else np.nan,
                tgt_ratio=r["target_" + f] / r["target_nat"] if r["target_nat"] > 0 else np.nan))
    d = pd.DataFrame(recs)
    pd.set_option("display.width", 220)
    print("circuits %d | members per circuit median %d | naturally active at anchor: median %.0f%%" % (
        d.seed.nunique(), d.n.median(), 100 * d.active.median()))
    cols = ["pre_med", "post_med", "restored", "above", "below", "dead", "switched_on", "mass_pre", "mass_post",
            "alpha_med", "comp_r", "tgt_ratio"]
    print("\nmedians over circuits, by ablation method:")
    print(d.groupby("fill")[cols].median().round(2).to_string())
    print("\nby ablation method x depth band:")
    print(d.groupby(["fill", "band"])[cols].median().round(2).to_string())
    print("\nby ablation method x pass:")
    print(d.groupby(["fill", "passes"])[cols].median().round(2).to_string())
    print("\nby ablation method x target site kind:")
    print(d.groupby(["fill", "kind"])[cols].median().round(2).to_string())


if __name__ == "__main__":
    report() if os.environ.get("REPORT") else main()
