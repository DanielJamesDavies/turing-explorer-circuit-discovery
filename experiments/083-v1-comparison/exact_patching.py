"""Exact activation patching as ground truth for AtP and IG (DAN-159, Daniel 2026-10-24), on a stratified subset.

Per target, on its v1 training contexts: the EXACT effect of each candidate latent on the target's pre-activation at
the anchor, Delta(s, i) = mean over contexts of [pre(latent i at site s set to its mean value, at every position;
everything else live) - pre(clean)]. The mean is the training-context mean (the A ablation value), the same one AtP
and IG start from, so AtP's score grad . (natural - mean) is exactly the first-order estimate of -Delta. The edit is
x_s <- x_s + W_dec[:, i] (mean_i - c_i), c_i = the latent's live code at that site (the SAE error is untouched); the
forward stops at the target's site.

Candidates: every latent active (post-Top-K > 0) at an upstream site at some position <= the anchor of some training
context (CAND=all), or, when that is more than CAND_MAX (default 6,000), the union of the top-2,000 by AtP, the top-2,000
by IG (both by |signed sum|, the SFC aggregate) and a seeded random 1,000 of the remaining active latents (CAND=subset);
the row records which and how many.

Per target it reports: Spearman correlation of AtP and IG (sfc, and max) with -Delta, signed and on |.|; top-n overlap
with the exact top-n by |Delta| at the W_0.001 size; and the exact-ranked circuit (global top-n by |Delta| at the
W_0.001 size) scored by driver.Runner.score at alpha = 1 and with fitted coefficients (attr_fitted.py's fit: gates
open, lambda 0, the production objective, contexts and 400 steps).

Targets: TARGETS_EXACT (default: N_EXACT = 10 drawn from the sweep list, seed 20261024, one per (depth band, site kind)
cell first, then the remaining draws from the deep band; written to exact/targets.txt).

  CUDA_VISIBLE_DEVICES=0 SHARD=0/1 PYTHONPATH=src python experiments/083-v1-comparison/exact_patching.py
  env: N_EXACT (10)  TARGETS_EXACT  CAND (auto | all | subset)  CAND_MAX (6000)  ROWS (64 patched rows per forward)
       IG_STEPS (10)  STEPS (400)  LIMIT  SHARD  OUT083  CTX_DIR (read by driver)  REPORT=1 (summary only, no model)
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
sys.path.insert(0, str(HERE))
import cmp_common as C  # noqa: E402

N_EXACT = int(os.environ.get("N_EXACT", 10))
CAND = os.environ.get("CAND", "auto")
CAND_MAX = int(os.environ.get("CAND_MAX", 6000))
ROWS = int(os.environ.get("ROWS", 64))
TOP_ATP = TOP_IG = 2000
N_RANDOM = 1000


def log(s):
    print("[%s %s] %s" % (time.strftime("%H:%M:%S"), C.TAG, s), flush=True)


def pick_targets(keys, n=N_EXACT, seed=20261024):
    """Stratified: one per (depth band, site kind) cell (cells in a fixed order), then the remaining draws from the
    deepest band; deterministic."""
    rng = random.Random(seed)
    cells = {}
    for k in keys:
        l, kind, _ = C.parse(k)
        cells.setdefault((C.depth_band(l), kind), []).append(k)
    order = [(b, k) for b, _, _ in C.DEPTH_BANDS for k in ("attn", "mlp", "resid")]
    out = [rng.choice(sorted(cells[c])) for c in order if c in cells]
    deep = sorted(k for k in keys if C.depth_band(C.parse(k)[0]) == C.DEPTH_BANDS[-1][0] and k not in out)
    out += rng.sample(deep, max(0, min(n - len(out), len(deep))))
    return out[:n]


class _Stop(Exception):
    pass


class PatchOne:
    """Row r of the batch: latent lat[r] at `site` set to its mean value at every position (all other latents and the
    SAE error untouched, every other site live); the target's pre-activation read at anchor[r] and the forward
    stopped at the target's site."""

    def __init__(self, bank, site, lats, means, wdec, seed, w_seed, b_seed, anchors):
        self.bank, self.site, self.lats, self.means, self.wdec = bank, site, lats, means, wdec
        self.seed, self.w_seed, self.b_seed, self.anchors = seed, w_seed, b_seed, anchors
        self.pre = None

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        s = (layer_idx, kind)
        if s == self.seed:
            pre = x @ self.w_seed.to(device=x.device, dtype=x.dtype) + self.b_seed.to(device=x.device, dtype=x.dtype)
            rr = torch.arange(pre.shape[0], device=pre.device)
            self.pre = pre[rr, self.anchors.to(pre.device).clamp(0, pre.shape[1] - 1)].float().detach()
            raise _Stop()
        if s != self.site or self.lats is None:
            return x
        ta, ti = self.bank.encode(x, kind, layer_idx)
        lat = self.lats.to(ti.device)
        c = (ta * (ti == lat[:, None, None])).sum(-1)                                   # [R, T] live code
        delta = self.means.to(c.device)[lat][:, None] - c                               # [R, T]
        return x + (delta[..., None] * self.wdec.to(x.device, x.dtype)[:, lat].T[:, None, :]).to(x.dtype)


def run_rows(G, patch, tokens):
    try:
        with torch.no_grad():
            G["inference"].forward(tokens, patcher=patch, return_activations=False, tokenize_final=False)
    except _Stop:
        pass
    return patch.pre


def active_in_prefix(G, sites, pos, arg, batch=16):
    """{site: bool [d_sae]}: latents with a nonzero post-Top-K code at some position <= the anchor of some context."""
    bank = G["bank"]
    k2i = {k: n for n, k in enumerate(G["KINDS"])}
    act = {s: torch.zeros(bank.d_sae, dtype=torch.bool) for s in sites}
    G["inference"].disable_compile()
    try:
        for s0 in range(0, int(pos.shape[0]), batch):
            an = arg[s0:s0 + batch]

            def hook(layer_idx, activations):
                for kd in G["KINDS"]:
                    s = (layer_idx, kd)
                    if s in act:
                        ta, ti = bank.encode(activations[k2i[kd]], kd, layer_idx)
                        keep = (torch.arange(ta.shape[1], device=ta.device)[None, :] <= an.to(ta.device)[:, None])
                        hit = ti[(ta > 0) & keep[..., None]]
                        act[s][hit.long().cpu()] = True
            with torch.no_grad():
                G["inference"].forward(pos[s0:s0 + batch], activations_callback=hook, return_activations=False,
                                       tokenize_final=False)
    finally:
        G["inference"].enable_compile()
    return act


def exact_effects(G, key, sites, cands, pos, arg, means):
    """Delta per candidate (CPU float64 [n]) and the clean anchor pre-activation mean."""
    bank = G["bank"]
    layer, kind, i = C.parse(key)
    sae = bank.saes[kind][layer]
    w_seed, b_seed = sae.encoder.weight[i].detach(), sae._get_bias_eff()[i].detach()
    B = int(pos.shape[0])
    clean = torch.cat([run_rows(G, PatchOne(bank, None, None, None, None, (layer, kind), w_seed, b_seed,
                                            arg[s0:s0 + ROWS]), pos[s0:s0 + ROWS]) for s0 in range(0, B, ROWS)]).cpu().double()
    eff = torch.zeros(len(cands), dtype=torch.float64)
    G["inference"].disable_compile()
    try:
        for s in sites:
            idx = [j for j, (ss, _) in enumerate(cands) if ss == s]
            if not idx:
                continue
            wdec = bank.saes[s[1]][s[0]].decoder.weight.detach()
            lats = torch.tensor([cands[j][1] for j in idx], dtype=torch.long)
            pairs = [(j, b) for j in range(len(idx)) for b in range(B)]           # (candidate, context) rows
            tot = torch.zeros(len(idx), dtype=torch.float64)
            t0 = time.time()
            for r0 in range(0, len(pairs), ROWS):
                if r0 and r0 % (ROWS * 500) == 0:
                    log("  %s site %s: %d / %d rows, %.0fs" % (key, s, r0, len(pairs), time.time() - t0))
                pj = torch.tensor([p[0] for p in pairs[r0:r0 + ROWS]])
                pb = torch.tensor([p[1] for p in pairs[r0:r0 + ROWS]])
                patch = PatchOne(bank, s, lats[pj], means[s], wdec, (layer, kind), w_seed, b_seed, arg[pb.to(arg.device)])
                pre = run_rows(G, patch, pos[pb.to(pos.device)]).cpu().double()
                tot.index_add_(0, pj, pre - clean[pb])
            eff[torch.tensor(idx)] = tot / B
    finally:
        G["inference"].enable_compile()
    return eff, float(clean.mean())


def check_edit(G, site, lat, means, pos):
    """The column edit equals bank.decode of the one-latent code change (linear decoder): checked once per target."""
    bank = G["bank"]
    d = torch.zeros(2, 3, bank.d_sae, device=pos.device)
    d[..., lat] = 1.0
    ref = bank.decode(d, site[1], site[0], add_bias=False)
    col = bank.saes[site[1]][site[0]].decoder.weight.detach()[:, lat].to(ref.device, ref.dtype)
    assert torch.allclose(ref, col.expand_as(ref), rtol=1e-3, atol=1e-4), "decoder is not the linear column map"


def spearman(a, b):
    import pandas as pd
    a, b = pd.Series(np.asarray(a, dtype=float)), pd.Series(np.asarray(b, dtype=float))
    return float(a.rank().corr(b.rank())) if len(a) > 2 else None


def main():
    if os.environ.get("REPORT") == "1":
        return report()
    driver = C.import_driver()
    import attr_baselines as A
    import attr_fitted as F
    from eval.floors import collect_site_means
    d = C.OUT / "exact"
    d.mkdir(parents=True, exist_ok=True)
    if os.environ.get("TARGETS_EXACT"):
        keys = [t for t in Path(os.environ["TARGETS_EXACT"]).read_text().split() if t]
    else:
        keys = pick_targets([t for t in C.TARGETS.read_text().split() if t])
        (d / "targets.txt").write_text("\n".join(keys) + "\n")
    keys = keys[C.SHARD_I::C.SHARD_K][:C.LIMIT] if C.LIMIT else keys[C.SHARD_I::C.SHARD_K]
    sizes = C.sizes_by_arm(C.sweep_rows())
    rows_path = d / ("rows.%s.jsonl" % C.TAG)
    done = C.done_keys(rows_path, ("seed", "variant", "ctx_dir"))
    log("exact patching | %d targets %s | cand %s (max %d) | rows %d" % (len(keys), keys, CAND, CAND_MAX, ROWS))
    F.install_support_wrapper()
    R = driver.Runner()
    G, ctx_dir = R.G, str(R.ctx_dir)
    M = R.method(driver.GAMMA, driver.LAM, True, steps=F.STEPS or driver.STEPS)
    fh, fr = open(rows_path, "a"), open(d / ("train.%s.jsonl" % C.TAG), "a")
    for n_, key in enumerate(keys):
        todo = [v for v in ("stats", "a1", "fit") if (key, v, ctx_dir) not in done]
        n_ref = sizes.get(key, {}).get(C.REF_ARM)
        if not todo or n_ref is None:
            continue
        rec = R.contexts(key)
        if rec.get("strong") is None:
            continue
        arm, thin, pos, arg, h = C.training_contexts(R, rec)
        base = dict(seed=key, method="exact_patch", ranking="exact", size_arm=C.REF_ARM, n_target=int(n_ref),
                    train_arm=arm, thin=thin, held="strong", shard=C.TAG, **C.provenance(R, h))
        try:
            path = d / "effects" / ("%s.%s.pt" % (key, h))
            if path.exists():
                E = torch.load(path, weights_only=False)
            else:
                t0 = time.time()
                layer, kind, i = C.parse(key)
                sites = A.upstream(G, key)
                means = collect_site_means(G["inference"], G["bank"], pos, set(sites))
                sae = G["bank"].saes[kind][layer]
                w_seed, b_seed = sae.encoder.weight[i].detach(), sae._get_bias_eff()[i].detach()
                scores, secs = {}, {}
                for name, alphas in (("atp", A.arm_alphas("atp")), ("abl_ig", A.arm_alphas("abl_ig"))):
                    t = time.time()
                    red = A.ig_reduce(G["inference"], G["bank"], sites, pos, arg, seed=(layer, kind), w_seed=w_seed,
                                      b_seed=b_seed, objective="drive", target_act=0.0, floors=means, alphas=alphas,
                                      batch_size=A.ATTR_BS)[0]
                    scores[name] = {rk: {s: red[s][rk].cpu() for s in sites} for rk in ("sfc", "max")}
                    secs[name] = round(time.time() - t, 1)
                act = active_in_prefix(G, sites, pos, arg)
                pool = [(s, int(j)) for s in sites for j in act[s].nonzero(as_tuple=True)[0].tolist()]
                mode = CAND if CAND != "auto" else ("all" if len(pool) <= CAND_MAX else "subset")
                if mode == "all":
                    cands = pool
                else:
                    def top(name, n):
                        flat = [(abs(float(scores[name]["sfc"][s][j])), s, j) for s, j in pool]
                        return {(s, j) for _, s, j in sorted(flat, key=lambda t: (-t[0], sites.index(t[1]), t[2]))[:n]}
                    keep = top("atp", TOP_ATP) | top("abl_ig", TOP_IG)
                    rest = sorted(set(pool) - keep, key=lambda t: (sites.index(t[0]), t[1]))
                    keep |= set(random.Random(C.key_seed(key)).sample(rest, min(N_RANDOM, len(rest))))
                    cands = sorted(keep, key=lambda t: (sites.index(t[0]), t[1]))
                check_edit(G, cands[0][0], cands[0][1], means, pos)
                t = time.time()
                eff, clean = exact_effects(G, key, sites, cands, pos, arg, {s: m.float() for s, m in means.items()})
                secs["exact"] = round(time.time() - t, 1)
                E = dict(key=key, sites=sites, cands=cands, effect=eff, clean_pre=clean, mode=mode, n_pool=len(pool),
                         scores={n: {rk: torch.tensor([float(v[rk][s][j]) for s, j in cands]) for rk in v}
                                 for n, v in scores.items()},
                         secs=secs, secs_total=round(time.time() - t0, 1), rows_per_forward=ROWS,
                         n_rows=len(cands) * int(pos.shape[0]))
                C.save_ranking_file(path, E)
            eff = E["effect"].numpy()
            order = sorted(range(len(eff)), key=lambda j: (-abs(eff[j]), j))
            top_exact = {E["cands"][j] for j in order[:n_ref]}
            members = {}
            for j in order[:n_ref]:
                members.setdefault(E["cands"][j][0], []).append(E["cands"][j][1])
            if "stats" in todo:
                st = dict(base, variant="stats", cand_mode=E["mode"], n_pool=E["n_pool"], n_cand=len(E["cands"]),
                          clean_pre=E["clean_pre"], secs=E["secs"], secs_total=E["secs_total"], n_rows=E["n_rows"],
                          exact_abs_sum=float(np.abs(eff).sum()))
                for name, v in E["scores"].items():
                    for rk, sc in v.items():
                        sc = sc.numpy()
                        st["spearman_%s_%s" % (name, rk)] = spearman(sc, -eff)
                        st["spearman_abs_%s_%s" % (name, rk)] = spearman(np.abs(sc), np.abs(eff))
                        tops = {E["cands"][j] for j in sorted(range(len(sc)), key=lambda j: (-abs(sc[j]), j))[:n_ref]}
                        st["overlap_%s_%s" % (name, rk)] = len(tops & top_exact) / max(1, n_ref)
                C.write_row(fh, st)
                f3 = lambda v: "-" if v is None else "%.3f" % v
                log("%2d/%d %-16s cand %d of %d (%s) | spearman AtP %s IG %s | |.| AtP %s IG %s | overlap AtP %s "
                    "IG %s | secs %s" % (n_ + 1, len(keys), key, len(E["cands"]), E["n_pool"], E["mode"],
                                         *(f3(st[k]) for k in ("spearman_atp_sfc", "spearman_abl_ig_sfc",
                                                               "spearman_abs_atp_sfc", "spearman_abs_abl_ig_sfc",
                                                               "overlap_atp_sfc", "overlap_abl_ig_sfc")), E["secs"]))
            todo_sc = [v for v in ("a1", "fit") if v in todo]
            final = d / "circuits" / ("%s.%s.pt" % (key, h))
            for row in F.score_variants(R, M, rec, key, arm, members, todo_sc, dict(base, n_support=C.n_members(members)),
                                        final, fr, dict(method="exact_patch")):
                C.write_row(fh, row)
                log("%2d/%d %-16s exact-ranked %-3s n=%s/%d Z %s A %s C %s nec %s" % (
                    n_ + 1, len(keys), key, row["variant"], row.get("n"), n_ref,
                    *("%.2f" % row[c] if isinstance(row.get(c), float) else "-" for c in C.HEAD + [C.NEC])))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            C.write_row(fh, dict(base, variant="stats", error="%s: %s" % (type(e).__name__, str(e)[:300])))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log("DONE")


def report():
    rows = C.last_by(C.ok_rows(C.read_jsonl(C.OUT / "exact" / "rows.*.jsonl")), ("seed", "variant"))
    st = [r for r in rows if r["variant"] == "stats"]
    lines = ["# Exact activation patching vs AtP / IG (%d targets)" % len(st), "",
             "| target | cand (of pool) | AtP ρ | IG ρ | AtP ρ abs | IG ρ abs | AtP overlap | IG overlap | exact-ranked "
             "Z/A/C α=1 | fitted | secs |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    sc = {(r["seed"], r["variant"]): r for r in rows if r["variant"] in ("a1", "fit")}
    f2 = lambda v: "—" if v is None else "%.2f" % v
    zac = lambda r: "—" if r is None else "/".join(f2(r.get(c)) for c in C.HEAD) + (" pass" if C.passes(r) else "")
    for r in st:
        lines.append("| %s | %d (%d, %s) | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["seed"], r["n_cand"], r["n_pool"], r["cand_mode"], f2(r["spearman_atp_sfc"]), f2(r["spearman_abl_ig_sfc"]),
            f2(r["spearman_abs_atp_sfc"]), f2(r["spearman_abs_abl_ig_sfc"]), f2(r["overlap_atp_sfc"]),
            f2(r["overlap_abl_ig_sfc"]), zac(sc.get((r["seed"], "a1"))), zac(sc.get((r["seed"], "fit"))),
            r["secs_total"]))
    if st:
        med = lambda k: np.nanmedian([r[k] for r in st if r.get(k) is not None])
        lines += ["", "Medians: Spearman (signed, sfc) AtP %.3f, IG %.3f; on |.| AtP %.3f, IG %.3f; top-n overlap AtP "
                  "%.2f, IG %.2f; max ranking: AtP ρ %.3f, IG ρ %.3f, overlap AtP %.2f, IG %.2f." % tuple(
                      med(k) for k in ("spearman_atp_sfc", "spearman_abl_ig_sfc", "spearman_abs_atp_sfc",
                                       "spearman_abs_abl_ig_sfc", "overlap_atp_sfc", "overlap_abl_ig_sfc",
                                       "spearman_atp_max", "spearman_abl_ig_max", "overlap_atp_max",
                                       "overlap_abl_ig_max"))]
    (C.OUT / "exact").mkdir(parents=True, exist_ok=True)
    (C.OUT / "exact" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
