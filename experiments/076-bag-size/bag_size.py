"""076 — how big is the bag? (the "overwhelmingly large bag of heuristics" critique)

Per target, rank the production circuit's members by their attribution to the target and score top-n truncations,
coefficients unchanged and no refit.

  attribution  A_j = alpha_j * d(mean target pre-activation at the anchors) / d alpha_j, in the circuit-only run under
               zero fill (Z), on the 48 strongest-TRAIN contexts (no held-out text is used to rank). This is the
               member's first-order contribution through every path inside the circuit.
  truncations  the top n members by |A_j|, n in NS (and all); scored on the 16 held-out strongest contexts under Z,
               A (activating mean, sparsity-preserving) and C (contrast mean, sparsity-preserving), each against its
               own empty circuit, the protocol-v1 ratio (x - e) / (a_pos - e), post-Top-K read.
  concentration the number of members holding 50 / 80 / 90 % of sum |A_j|.

Targets: the same 747 random-null targets as 073, cells interleaved. Nothing in out_full is written.
The all-15k concentration from the explorer bundle's magnitude shares (|alpha| x anchor x decoder norm) is in
report.py (CPU).

  PYTHONPATH=src python experiments/076-bag-size/bag_size.py      env: LIMIT  OUT76  REPORT=1
"""
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "073-strict-eval"))
import strict_eval as SE  # noqa: E402  (imports 062's driver, sets OUT / EVAL_SPLIT)

LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
OUT76 = Path(os.environ.get("OUT76", str(HERE / "out")))
NS = (1, 2, 5, 10, 20, 50, 100, 200, 500)
ATTR_BATCH = int(os.environ.get("ATTR_BATCH", 4))       # backward keeps a dense [B, T, d_sae] code per site


def attributions(G, key, rec_probe, layer, kd, sl, w_seed, b_seed, UPS, kt, sc):
    """A_j per member (site-major order of kt), gradient through the circuit-only Z run on the train contexts."""
    from eval.wcm_eval import TapCircuitOnly
    inference, bank = G["inference"], G["bank"]
    D = int(bank.d_sae)
    pt, pa, n_tr = rec_probe
    toks, anc = pt[:n_tr], pa[:n_tr]
    params = {s: sc[s].detach().clone().float().requires_grad_(True) for s in kt}
    grads = {s: torch.zeros_like(params[s]) for s in kt}
    inference.disable_compile()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(toks.shape[0]), ATTR_BATCH):
                tk = toks[s0:s0 + ATTR_BATCH]
                B = int(tk.shape[0])
                ks = {}
                for s, t in kt.items():
                    base = torch.ones(D, device=t.device)
                    ks[s] = base.index_put((t,), params[s])
                p = TapCircuitOnly(bank=bank, keep_indices={s: set(t.tolist()) for s, t in kt.items()}, in_scope=UPS,
                                   seed_layer=layer, seed_kind=kd, seed_latent_idx=sl, pos_argmax=anc[s0:s0 + B],
                                   keep_tensors=kt, keep_scales=ks, w_seed=w_seed, b_seed=b_seed)
                with torch.enable_grad():
                    inference.forward(tk, patcher=p, grad_enabled=True, return_activations=False, tokenize_final=False)
                    pre = p.seed_pre
                    rr = torch.arange(B, device=pre.device)
                    y = pre[rr, anc[s0:s0 + B].to(pre.device).clamp(0, pre.shape[1] - 1)].float().sum() / toks.shape[0]
                    gs = torch.autograd.grad(y, [params[s] for s in kt], allow_unused=True)
                for s, gr in zip(kt, gs):
                    if gr is not None:
                        grads[s] += gr.detach()
    finally:
        inference.enable_compile()
    return {s: (params[s].detach() * grads[s]) for s in kt}


def score_target(G, key):
    from circuit.protocol_contexts import eval_probe
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_anchors, collect_site_means
    from eval.wcm_eval import NaturalTap, TapCircuitOnly, circuit_alphas, contrast_split, site_encoder
    inference, bank, dev = G["inference"], G["bank"], G["device"]
    K, D = int(bank.k), int(bank.d_sae)
    rec = G["R"].contexts(key)
    probe = eval_probe(rec, "strong", dev)
    n_tr = int(probe.metadata["n_train"])
    pt, pa = probe.pos_tokens[:64], probe.pos_argmax[:64]
    pt_tr, pa_tr, pt_ho, pa_ho = pt[:n_tr], pa[:n_tr], pt[n_tr:], pa[n_tr:]
    nt = probe.neg_tokens[:64]
    nt_tr = nt[:contrast_split(int(nt.shape[0]), probe.metadata)]     # 086 rebuilt contrast: n_neg_train
    l, kd, i = key.split(".")
    layer, sl = int(l), int(i)
    site = (layer, kd)
    W, b = site_encoder(bank, layer, kd)
    w_seed, b_seed = W[sl], b[sl]
    UPS = set(upstream_sites(bank, layer, kd))
    c = torch.load(SE.MAIN_DIR / ("%s.pt" % key), weights_only=False)
    al = circuit_alphas(c)
    kt = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in al.items() if d}
    sc = {s: torch.tensor([al[s][int(j)] for j in kt[s].tolist()], device=dev) for s in kt}
    n_mem = sum(int(t.numel()) for t in kt.values())
    row = dict(seed=key, layer=layer, kind=kd, band=SE.band(layer), n=n_mem)
    if n_mem == 0:
        row["skip"] = "empty"
        return row

    A = attributions(G, key, (pt, pa, n_tr), layer, kd, sl, w_seed, b_seed, UPS, kt, sc)
    flat = [(float(A[s][j]), s, int(kt[s][j])) for s in kt for j in range(kt[s].numel())]
    flat.sort(key=lambda t: -abs(t[0]))
    mags = torch.tensor([abs(a) for a, _, _ in flat], dtype=torch.float64)
    tot = float(mags.sum())
    cum = (mags.cumsum(0) / tot) if tot > 0 else mags
    for q in (0.5, 0.8, 0.9):
        row["n%d" % int(q * 100)] = int((cum < q).sum()) + 1 if tot > 0 else None
    row["attr_sum_signed"] = float(sum(a for a, _, _ in flat))
    row["attr_sum_abs"] = tot
    row["frac_neg_attr"] = float(sum(abs(a) for a, _, _ in flat if a < 0) / tot) if tot > 0 else None
    row["top10_attr"] = [(round(a, 4), "%d.%s.%d" % (s[0], s[1], j)) for a, s, j in flat[:10]]

    # natural a_pos on held-out, ablation values
    nat = SE.run_full(inference, lambda a, b_: NaturalTap(bank, site, w_seed, b_seed, sl), pt_ho)
    a_pos = float(SE.at(nat, pa_ho).mean())
    row["a_pos"] = a_pos
    means_tr, _ = collect_site_anchors(inference, bank, pt_tr, UPS, pa_tr, pin_position_specific=False)
    means_neg = collect_site_means(inference, bank, nt_tr, UPS) if UPS else {}
    fills = {"Z": (None, False), "A": (means_tr, True), "C": (means_neg, True)}

    def co(keep_list, fill):
        mm, rtk = fills[fill]
        kk = {}
        for _, s, j in keep_list:
            kk.setdefault(s, []).append(j)
        ktt = {s: torch.tensor(sorted(v), device=dev, dtype=torch.long) for s, v in kk.items()}
        ks = {}
        for s, t in ktt.items():
            sv = torch.ones(D, device=dev)
            sv[t] = torch.tensor([al[s][int(j)] for j in t.tolist()], device=dev)
            ks[s] = sv
        tk = SE.run_full(inference, lambda a, b_: TapCircuitOnly(
            bank=bank, keep_indices={s: set(t.tolist()) for s, t in ktt.items()}, in_scope=UPS, seed_layer=layer,
            seed_kind=kd, seed_latent_idx=sl, pos_argmax=pa_ho[a:b_], site_means=mm, respect_topk=rtk, topk=K,
            keep_tensors=ktt, keep_scales=ks or None, w_seed=w_seed, b_seed=b_seed), pt_ho)
        return float(SE.at(tk, pa_ho).mean())

    empt = {f: co([], f) for f in fills}
    for f in fills:
        row["e_%s" % f] = empt[f]
    ns = sorted(set([n for n in NS if n < n_mem] + [n_mem]))
    trunc = {}
    for n in ns:
        sub = flat[:n]
        trunc[n] = {f: SE.ratio(co(sub, f), empt[f], a_pos) for f in fills}
    row["trunc"] = {str(n): v for n, v in trunc.items()}
    # smallest n (of the grid) at which all three are in the protocol band [0.8, 1.5]
    ok = [n for n in ns if all(v is not None and 0.8 <= v <= 1.5 for v in trunc[n].values())]
    row["n_min_band"] = min(ok) if ok else None
    return row


def main():
    OUT76.mkdir(parents=True, exist_ok=True)
    path = OUT76 / "rows.jsonl"
    keys = SE.order_targets(SE.default_targets())
    if LIMIT:
        keys = keys[:LIMIT]
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "error" not in r:
                done.add(r["seed"])
    R = SE.driver.Runner()
    G = dict(R.G)
    G["R"] = R
    print("076 bag size: %d targets, %d done" % (len(keys), len(done)), flush=True)
    fh = open(path, "a")
    t0, n_new = time.time(), 0
    for n_, key in enumerate(keys):
        if key in done:
            continue
        ts = time.time()
        try:
            row = score_target(G, key)
        except Exception as e:  # noqa: BLE001
            row = dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))
            traceback.print_exc()
        row["secs"] = round(time.time() - ts, 1)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        n_new += 1
        left = sum(1 for k in keys[n_ + 1:] if k not in done)
        print("[%s] %4d/%d %-16s %s n=%s n80=%s nmin=%s  %.1fs  ~%.0f min left" % (
            time.strftime("%H:%M"), n_ + 1, len(keys), key, row.get("error", "ok"), row.get("n"), row.get("n80"),
            row.get("n_min_band"), row["secs"], left * (time.time() - t0) / n_new / 60), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    if os.environ.get("REPORT"):
        import report
        report.main()
    else:
        main()
