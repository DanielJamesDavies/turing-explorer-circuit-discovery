"""073 + 074 — stricter evaluation of the 15k-run circuits (no refitting).

073, resample ablation (the causal-scrubbing critique). Every latent outside the circuit, at every upstream site, takes
its NATURAL code from a different real context (the donor) at the same position, instead of a mean or zero. Donor sets:
  A  the target's 48 strongest-train activating contexts (the resample counterpart of the A mean)
  C  the target's 48 train contrast contexts (the counterpart of the C mean)
  R  random corpus contexts (one fixed pool, a per-target draw)
Sparsity-preserving: members keep their alpha x live values; the donor's own active non-member latents fill the
remaining Top-K budget per position, largest first (the resample analogue of the *_topk fill). The SAE error of the
current stream is preserved, as in the engine's CircuitOnlyPatcher. Score: (x - e) / (a_pos - e) at the held-out
anchors, x = the circuit, e = the empty circuit under the same donors, means over N_DRAWS donor draws.

074, all-position simulation (the Gao critique). The circuit-only run's target read at EVERY position of the 16
held-out activating AND the 16 held-out contrast contexts, against the natural read: Pearson r, fraction of variance
explained, firing precision / recall, false firing on contrast contexts. Fills: Z (zero) and C (contrast mean,
sparsity-preserving: freeN_topk's fill).

Variants per target: the production circuit at its fitted alpha ("wcm"), the same members at alpha = 1 ("a1"), the
empty circuit ("empty"), and the target's random-null circuit with its fitted alpha ("rand"; 062 out_random_scale).
Targets: the 747 random-null targets (paired baseline for every target). Nothing in out_full is written.

  PYTHONPATH=src python experiments/073-strict-eval/strict_eval.py
  env: N_DRAWS (2)  LIMIT  TARGETS (default: the random-null targets)  OUT73 (default experiments/073-strict-eval/out)
       SELFTEST=1 (identity checks on 3 targets, then exit)   REPORT=1 (summary; no GPU)
       RAND_DIR (the random-null circuits, <key>.d0.pt; default 062 out_random_scale/circuits)
       SHARD=i/k (keys[i::k]; rows to rows.shard<i>.jsonl when k > 1; the random pool is shared, written once)
       FAMILY / DATA / STORES as 062 driver.py (public family: Runner on its bank, donor pool from its own corpus)
"""
import contextlib
import io
import json
import os
import random
import sys
import time
import traceback
import zlib
from pathlib import Path

import torch

HERE = Path(__file__).parent
EXP = HERE.parent
E62 = EXP / "062-h100-protocol-v1"
os.environ.setdefault("OUT", str(E62 / "out_full" / "out"))
os.environ["EVAL_SPLIT"] = "explicit"
sys.path.insert(0, str(E62))
import driver  # noqa: E402

N_DRAWS = int(os.environ.get("N_DRAWS", 2))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
OUT73 = Path(os.environ.get("OUT73", str(HERE / "out")))
RAND_DIR = Path(os.environ.get("RAND_DIR", str(E62 / "out_random_scale" / "circuits")))
MAIN_DIR = Path(os.environ["OUT"]) / "main" / "circuits"
BATCH = 16
POOL_N = 512
POOL_SEED = 20261016
VARIANTS = ("empty", "wcm", "a1", "rand")


def band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


# ----------------------------------------------------------------------------------------------- patchers
class ResamplePatcher:
    """Circuit-only run with a resample fill: members at alpha x live value; every other latent at every in-scope site
    from the donor's natural sparse code at the same position (non-member donor entries, largest first, up to the
    Top-K budget left after the active members). Error term of the current stream preserved. Taps the target read."""

    def __init__(self, bank, in_scope, seed_site, w_seed, b_seed, seed_idx, keep_t, scales, donor, K,
                 frac=1.0, pins=None, gen_seed=0):
        self.bank, self.in_scope, self.seed_site = bank, in_scope, seed_site
        self.w_seed, self.b_seed, self.seed_idx = w_seed, b_seed, seed_idx
        self.keep_t, self.scales, self.donor, self.K = keep_t, scales, donor, int(K)
        self.frac, self.pins, self.gen_seed = float(frac), pins, int(gen_seed)   # diagnostics (diag.py)
        self.seed_pre = None
        self.tap_tk = None

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        from eval.wcm_eval import _tap
        from sae.dense import sparse_topk_to_dense
        site = (layer_idx, kind)
        if site == self.seed_site:
            self.seed_pre = _tap(self, self.bank, x, site)
        if site not in self.in_scope:
            return x
        bank = self.bank
        ta, ti = bank.encode(x, kind, layer_idx)
        D = int(bank.d_sae)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        error = x - bank.decode(dense, kind, layer_idx)
        patched = torch.zeros_like(dense)
        kt = self.keep_t.get(site)
        if kt is not None:
            if self.pins is not None:
                vals = self.pins[site][:x.shape[0]].to(dense.dtype)
            elif getattr(self, "member_pre", False):       # diag: members read uncensored, relu(w.x + b)
                sae = bank.saes[kind][layer_idx]
                Wm = sae.encoder.weight[kt].to(x.dtype)
                bm = sae._get_bias_eff()[kt].to(x.dtype)
                vals = torch.relu(x @ Wm.T + bm).to(dense.dtype)
            else:
                vals = dense[:, :, kt]
            sv = self.scales.get(site)
            if sv is not None:
                vals = vals * sv.to(vals.dtype)
            patched[:, :, kt] = vals
        dta, dti = self.donor[site]
        B = int(x.shape[0])
        dta, dti = dta[:B].to(dense.dtype), dti[:B]
        valid = dta > 0
        if kt is not None:
            valid = valid & ~torch.isin(dti, kt)
        if self.frac < 1.0:                       # keep a random fraction of the donor entries, zero the rest
            g = torch.Generator(device=dta.device).manual_seed(self.gen_seed + 1000 * layer_idx + len(kind))
            valid = valid & (torch.rand(dta.shape, generator=g, device=dta.device) < self.frac)
        budget = (self.K - (patched != 0).sum(-1)).clamp(min=0)                     # [B, T]
        order = dta.masked_fill(~valid, float("-inf")).argsort(-1, descending=True)
        dta_s, dti_s, val_s = dta.gather(-1, order), dti.gather(-1, order), valid.gather(-1, order)
        rank = val_s.long().cumsum(-1) - 1
        use = val_s & (rank < budget.unsqueeze(-1))
        patched.scatter_add_(-1, dti_s, dta_s * use.to(dta_s.dtype))
        return bank.decode(patched, kind, layer_idx) + error


# ----------------------------------------------------------------------------------------------- helpers
def run_full(inference, make, tokens, pre=False):
    """Target read (post-Top-K) at every position: [n, T]; make(start, stop) builds the patcher for one batch.
    pre=True also returns relu(w.x + b), the uncensored pre-activation read: (tk, pre)."""
    out, outp = [], []
    inference.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(tokens.shape[0]), BATCH):
                tk = tokens[s0:s0 + BATCH]
                p = make(s0, s0 + int(tk.shape[0]))
                inference.forward(tk, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
                out.append(p.tap_tk.float())
                outp.append(torch.relu(p.seed_pre.float()))
    finally:
        inference.enable_compile()
    return (torch.cat(out), torch.cat(outp)) if pre else torch.cat(out)


def capture_codes(inference, bank, tokens, sites):
    """Natural sparse codes {site: (acts [n, T, k], idx [n, T, k])} at the given sites."""
    kinds = list(bank.kinds)
    k2i = {k: i for i, k in enumerate(kinds)}
    parts = {s: ([], []) for s in sites}

    def hook(layer_idx, acts):
        for kd in kinds:
            s = (layer_idx, kd)
            if s in parts:
                ta, ti = bank.encode(acts[k2i[kd]], kd, layer_idx)
                parts[s][0].append(ta.detach().float()); parts[s][1].append(ti.detach())

    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), BATCH):
                inference.forward(tokens[s0:s0 + BATCH], activations_callback=hook, return_activations=False,
                                  tokenize_final=False)
    finally:
        inference.enable_compile()
    return {s: (torch.cat(a), torch.cat(i)) for s, (a, i) in parts.items()}


def at(tk, anchors):
    rr = torch.arange(tk.shape[0], device=tk.device)
    return tk[rr, anchors.to(tk.device).clamp(0, tk.shape[1] - 1)]


def sim_metrics(c, n, n_act):
    """All-position agreement of circuit read c with natural read n ([N, T]; first n_act rows activating)."""
    c, n = c.flatten().double(), n.flatten().double()
    m = {}
    vc, vn = c - c.mean(), n - n.mean()
    den = float((vc.norm() * vn.norm()))
    m["r"] = float((vc * vn).sum() / den) if den > 1e-12 else None
    ss = float((vn ** 2).sum())
    m["fve"] = 1.0 - float(((c - n) ** 2).sum()) / ss if ss > 1e-12 else None
    cf, nf = c > 0, n > 0
    m["prec"] = float((cf & nf).sum() / cf.sum()) if cf.sum() > 0 else None
    m["rec"] = float((cf & nf).sum() / nf.sum()) if nf.sum() > 0 else None
    m["fp_rate_off"] = float((cf & ~nf).sum() / (~nf).sum()) if (~nf).sum() > 0 else None
    m["fp_mass"] = float(c[~nf].clamp(min=0).sum() / n[nf].sum()) if n[nf].sum() > 0 else None
    k = n_act * 64
    cc, nc = c[k:], n[k:]
    m["ctr_fire_rate"] = float((cc > 0).double().mean()) if cc.numel() else None
    m["ctr_nat_fire_rate"] = float((nc > 0).double().mean()) if nc.numel() else None
    ca, na = c[:k], n[:k]
    va, wa = ca - ca.mean(), na - na.mean()
    d2 = float(va.norm() * wa.norm())
    m["r_act"] = float((va * wa).sum() / d2) if d2 > 1e-12 else None
    return m


def ratio(x, e, ap):
    return (x - e) / (ap - e) if ap - e > 1e-9 else None


def random_pool(loader):
    from circuit.protocol_contexts import _load
    ranges = [(a, b) for a, b in loader.shard_id_ranges if a >= 0]
    rng = random.Random(POOL_SEED)
    ids = []
    while len(ids) < POOL_N:
        a, b = rng.choice(ranges)
        ids.append(rng.randint(a, b))
    pos, _, got = _load(loader, ids)
    keep = [i for i in range(pos.shape[0]) if int((pos[i] != 0).sum()) >= 60]
    return pos[keep][:POOL_N], [got[i] for i in keep][:POOL_N]


def default_targets():
    return sorted(p.name[:-len(".d0.pt")] for p in RAND_DIR.glob("*.d0.pt"))


def order_targets(keys):
    """Cells (band x kind) interleaved, so a partial run is still stratified."""
    from collections import defaultdict
    cells = defaultdict(list)
    for k in keys:
        l, kd, _ = k.split(".")
        cells[(band(int(l)), kd)].append(k)
    rng = random.Random(20261016)
    for c in cells.values():
        rng.shuffle(c)
    m = max(len(c) for c in cells.values())
    return [cells[c][i] for i in range(m) for c in sorted(cells) if i < len(cells[c])]


# ----------------------------------------------------------------------------------------------- one target
def score_target(G, key, pool, rng_seed=0, selftest=False, c_held=False):
    """c_held=True: the C donors are the 16 HELD-OUT contrast contexts instead of the 48 train ones (079: circuits
    trained against train-contrast donors must be scored on donors they never saw)."""
    from circuit.protocol_contexts import eval_probe
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_means
    from eval.wcm_eval import NaturalTap, TapCircuitOnly, circuit_alphas, contrast_split, site_encoder
    inference, bank, dev = G["inference"], G["bank"], G["device"]
    K, D = int(G["K"]), int(bank.d_sae)                  # K = the dictionary's Top-K k (128 / 32 / 16)
    R_ = G["R"]
    rec = R_.contexts(key)
    probe = eval_probe(rec, "strong", dev)
    n_tr = int(probe.metadata["n_train"])
    pt, pa = probe.pos_tokens[:64], probe.pos_argmax[:64]
    pt_tr, pt_ho, pa_ho = pt[:n_tr], pt[n_tr:], pa[n_tr:]
    nt = probe.neg_tokens[:64]
    n_ntr = contrast_split(int(nt.shape[0]), probe.metadata)          # 086 rebuilt contrast: n_neg_train
    nt_tr, nt_ho = nt[:n_ntr], nt[n_ntr:]
    l, kd, i = key.split(".")
    layer, sl = int(l), int(i)
    site = (layer, kd)
    W, b = site_encoder(bank, layer, kd)
    w_seed, b_seed = W[sl], b[sl]
    UPS = set(upstream_sites(bank, layer, kd))

    circ = {"wcm": torch.load(MAIN_DIR / ("%s.pt" % key), weights_only=False)}
    rp = RAND_DIR / ("%s.d0.pt" % key)
    if rp.exists():
        circ["rand"] = torch.load(rp, weights_only=False)
    specs = {"empty": ({}, {})}
    for name, c in circ.items():
        if c is None:
            continue
        al = circuit_alphas(c)
        kt = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in al.items() if d}
        sc = {s: torch.tensor([al[s][int(j)] for j in kt[s].tolist()], device=dev) for s in kt}
        specs[name] = (kt, sc)
        if name == "wcm":
            specs["a1"] = (kt, {})

    # natural reads
    def nat(tokens):
        return run_full(inference, lambda a, b_: NaturalTap(bank, site, w_seed, b_seed, sl), tokens)
    nat_act, nat_ctr = nat(pt_ho), nat(nt_ho)
    a_pos = float(at(nat_act, pa_ho).mean())
    row = dict(seed=key, layer=layer, kind=kd, band=band(layer), a_pos=a_pos, n_up=len(UPS),
               n_wcm=sum(len(v) for v in specs["wcm"][0].values()) if "wcm" in specs else None,
               has_rand="rand" in specs)

    # ---- 074: all-position, Z and C (sparsity-preserving contrast mean)
    means_neg = collect_site_means(inference, bank, nt_tr, UPS) if UPS else {}
    both = torch.cat([pt_ho, nt_ho])
    nat_both = torch.cat([nat_act, nat_ctr])
    anc_both = torch.cat([pa_ho, torch.zeros(nt_ho.shape[0], dtype=pa_ho.dtype, device=pa_ho.device)])
    for fill, means, rtk in (("Z", None, False), ("C", means_neg, True)):
        for v, (kt, sc) in specs.items():
            keep_idx = {s: set(t.tolist()) for s, t in kt.items()}
            ks = None
            if sc:
                ks = {}
                for s, t in kt.items():
                    sv = torch.ones(D, device=dev)
                    sv[t] = sc[s]
                    ks[s] = sv
            c_both = run_full(inference, lambda a, b_, _k=keep_idx, _kt=kt, _ks=ks, _m=means, _r=rtk: TapCircuitOnly(
                bank=bank, keep_indices=_k, in_scope=UPS, seed_layer=layer, seed_kind=kd, seed_latent_idx=sl,
                pos_argmax=anc_both[a:b_], site_means=_m, respect_topk=_r, topk=K, keep_tensors=_kt,
                keep_scales=_ks, w_seed=w_seed, b_seed=b_seed), both)
            for mk, mv in sim_metrics(c_both, nat_both, int(pt_ho.shape[0])).items():
                row["ap_%s_%s_%s" % (fill, v, mk)] = mv
            row["ap_%s_%s_anchor_mean" % (fill, v)] = float(at(c_both[:pt_ho.shape[0]], pa_ho).mean())

    # ---- 073: resample fills
    donor_sets = {"A": pt_tr, "C": nt_ho if c_held else nt_tr}
    g = torch.Generator().manual_seed(rng_seed + zlib.crc32(key.encode()))
    ridx = torch.randperm(pool.shape[0], generator=g)[:BATCH * N_DRAWS]
    donor_sets["R"] = pool[ridx].to(dev)
    sites = sorted(UPS)
    for ds, toks in donor_sets.items():
        codes = capture_codes(inference, bank, toks, sites) if sites else {}
        n_d = int(toks.shape[0])
        acc = {v: [] for v in specs}
        donor_nat = []
        for d in range(N_DRAWS):
            if ds == "R":
                sel = torch.arange(d * BATCH, (d + 1) * BATCH)
            else:
                sel = torch.randperm(n_d, generator=g)[:BATCH]
            sel = sel[:int(pt_ho.shape[0])]
            donor = {s: (codes[s][0][sel.to(codes[s][0].device)], codes[s][1][sel.to(codes[s][1].device)])
                     for s in sites}
            donor_nat.append(float(at(nat(toks[sel.to(toks.device)]), pa_ho[:len(sel)]).mean()))
            for v, (kt, sc) in specs.items():
                svec = {s: sc[s] for s in kt} if sc else {}
                tk = run_full(inference, lambda a, b_, _kt=kt, _sv=svec, _d=donor: ResamplePatcher(
                    bank, UPS, site, w_seed, b_seed, sl, _kt, _sv, {s: (x[a:b_], y[a:b_]) for s, (x, y) in _d.items()},
                    K), pt_ho[:len(sel)])
                acc[v].append(float(at(tk, pa_ho[:len(sel)]).mean()))
        e = sum(acc["empty"]) / N_DRAWS
        row["rs_%s_empty_raw" % ds] = e
        row["rs_%s_leak" % ds] = e / a_pos if a_pos > 1e-9 else None
        row["rs_%s_donor_nat" % ds] = sum(donor_nat) / N_DRAWS
        for v in specs:
            if v == "empty":
                continue
            x = sum(acc[v]) / N_DRAWS
            row["rs_%s_%s_raw" % (ds, v)] = x
            row["rs_%s_%s" % (ds, v)] = ratio(x, e, a_pos)
            row["rs_%s_%s_draws" % (ds, v)] = [ratio(xi, ei, a_pos) for xi, ei in zip(acc[v], acc["empty"])]
        del codes

    if selftest:
        # identity 1: self-donor, empty circuit == natural; identity 2: self-donor, alpha = 1 == natural;
        # identity 3: all-zero donor == the engine's zero fill (free0) for the production circuit
        codes = capture_codes(inference, bank, pt_ho, sites)
        out = {}
        for v in ("empty", "a1"):
            kt, sc = specs[v]
            tk = run_full(inference, lambda a, b_, _kt=kt: ResamplePatcher(
                bank, UPS, site, w_seed, b_seed, sl, _kt, {}, {s: (x[a:b_], y[a:b_]) for s, (x, y) in codes.items()},
                K), pt_ho)
            out["self_%s_maxdiff" % v] = float((tk - nat_act).abs().max())
            out["self_%s_anchor" % v] = (float(at(tk, pa_ho).mean()), a_pos)
            # the engine's own path with every latent kept (the same decode + error round trip): roundoff reference
            allk = {s: torch.arange(D, device=dev) for s in sites}
            tk_e = run_full(inference, lambda a, b_, _kt=allk: TapCircuitOnly(
                bank=bank, keep_indices={s: set() for s in sites}, in_scope=UPS, seed_layer=layer, seed_kind=kd,
                seed_latent_idx=sl, pos_argmax=pa_ho[a:b_], keep_tensors=_kt, w_seed=w_seed, b_seed=b_seed), pt_ho)
            out["engine_keepall_maxdiff"] = float((tk_e - nat_act).abs().max())
        zc = {s: (torch.zeros_like(x), y) for s, (x, y) in codes.items()}
        kt, sc = specs["wcm"]
        tk = run_full(inference, lambda a, b_: ResamplePatcher(
            bank, UPS, site, w_seed, b_seed, sl, kt, {s: sc[s] for s in kt},
            {s: (x[a:b_], y[a:b_]) for s, (x, y) in zc.items()}, K), pt_ho)
        out["zero_vs_Z_anchor"] = (float(at(tk, pa_ho).mean()), row["ap_Z_wcm_anchor_mean"])
        row["selftest"] = out
    return row


# ----------------------------------------------------------------------------------------------- main
def main():
    OUT73.mkdir(parents=True, exist_ok=True)
    selftest = os.environ.get("SELFTEST") == "1"
    path = OUT73 / "selftest.jsonl" if selftest else driver.shard_jsonl(OUT73, "rows")
    keys = [t for t in Path(os.environ["TARGETS"]).read_text().split() if t] if os.environ.get("TARGETS") \
        else order_targets(default_targets())
    if selftest:
        keys = keys[:3]
    elif LIMIT:
        keys = keys[:LIMIT]
    keys = keys[driver.SHARD_I::driver.SHARD_K]
    done = set()
    if not selftest:
        for r in driver.read_jsonl_all(OUT73, "rows"):
            if "error" not in r:
                done.add(r["seed"])
    R = driver.Runner()
    G = dict(R.G)
    G["R"] = R
    pool_path = OUT73 / "random_pool.pt"
    if pool_path.exists():
        pool = torch.load(pool_path)["tokens"]
    else:
        # the run's (family's) own corpus; deterministic (POOL_SEED), so shards racing here write the same pool
        pool, ids = random_pool(G["M0"].probe_builder.loader)
        tmp = pool_path.with_name("random_pool.%s.tmp" % driver.TAG)
        torch.save(dict(tokens=pool, ids=ids, seed=POOL_SEED), tmp)
        os.replace(tmp, pool_path)
    pool = pool.to(G["device"])
    print("073/074 strict eval: %d targets, %d done, pool %d (shard %d/%d)" % (
        len(keys), len(done), pool.shape[0], driver.SHARD_I, driver.SHARD_K), flush=True)
    fh = open(path, "a")
    t0 = time.time()
    n_new = 0
    for n_, key in enumerate(keys):
        if key in done:
            continue
        ts = time.time()
        try:
            row = score_target(G, key, pool, selftest=selftest)
        except Exception as e:  # noqa: BLE001
            row = dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))
            traceback.print_exc()
        row["secs"] = round(time.time() - ts, 1)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        n_new += 1
        rate = (time.time() - t0) / n_new
        left = sum(1 for k in keys[n_ + 1:] if k not in done)
        print("[%s] %4d/%d %-16s %s  %.1fs  ~%.0f min left" % (time.strftime("%H:%M"), n_ + 1, len(keys), key,
                                                           row.get("error", "ok"), row["secs"], left * rate / 60),
              flush=True)
        if selftest:
            print(json.dumps(row.get("selftest")), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    if os.environ.get("REPORT"):
        import report
        report.main()
    else:
        main()
