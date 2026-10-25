"""075 — does the circuit preserve what the target DOES downstream? (reader-recovered and KL-recovered)

Leo Gao's standard (judge a simulation by its downstream effect, KL rather than correlation), applied where our
targets act: on other latents. Per target:

  readers    the N_READ downstream latents (every site after the target's in causal order) whose summed activation
             changes most when the target is zeroed in the live stream, chosen on the 48 strongest-TRAIN contexts;
             sign_j = the sign of that change.
  splice     on the 16 held-out strongest contexts, set the target latent, at every position, to the value a circuit-
             only run computes for it (all other latents live), then read the readers and the logits.
             Values from: wcm under C (contrast mean, sparsity-preserving; the protocol's freeN_topk run), wcm under Z,
             the empty circuit under C, the random-null circuit under C, and the target's own natural value
             ("nat", sanity: must recover ~1).
  reader_rec = sum_j sign_j (r_splice - r_ablate) / sum_j sign_j (r_clean - r_ablate), sums over positions.
  kl_rec     = 1 - KL(clean || splice) / KL(clean || ablate), mean over positions; reported with KL(clean || ablate).

Targets: the 747 random-null targets (073's order). Nothing in out_full is written.

  PYTHONPATH=src python experiments/075-reader-recovered/reader_recovered.py     env: LIMIT  N_READ (20)  OUT75  REPORT=1
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
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "073-strict-eval"))
import strict_eval as SE  # noqa: E402

LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
N_READ = int(os.environ.get("N_READ", 20))
OUT75 = Path(os.environ.get("OUT75", str(HERE / "out")))


class SpliceRead:
    """Live stream. At the target site: the target latent set to ``value`` [B, T] (None: no edit; "zero": 0),
    SAE error preserved. At downstream sites: accumulate per-latent activation sums (``sums``) or read named latents
    (``read``: {site: idx tensor} -> self.reads[site] [B, T, n])."""

    def __init__(self, bank, site, sl, value, down_sites, mode="sums", read=None, dec=None):
        self.bank, self.site, self.sl, self.value = bank, site, sl, value
        self.down, self.mode, self.read = set(down_sites), mode, read or {}
        self.sums, self.reads = {}, {}
        self.dec = dec                     # the target latent's decoder direction [d_model] (decode(e_sl) - decode(0))

    def __call__(self, model):
        from model.hooks import multi_patch
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        from sae.dense import sparse_topk_to_dense
        s = (layer_idx, kind)
        bank = self.bank
        if s == self.site and self.value is not None:
            # exact additive edit: only the target's own contribution changes (no full decode round trip, whose
            # bf16 roundoff is comparable to a weak target's effect)
            from sae.dense import target_latent_activations
            ta, ti = bank.encode(x, kind, layer_idx)
            cur = target_latent_activations(ta, ti, self.sl).float()                      # [B, T]
            B = int(x.shape[0])
            new = torch.zeros_like(cur) if isinstance(self.value, str) else self.value[:B].float()
            return (x.float() + (new - cur)[..., None] * self.dec.float()).to(x.dtype)
        if s in self.down:
            ta, ti = bank.encode(x, kind, layer_idx)
            if self.mode == "sums":
                dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=torch.float32)
                self.sums[s] = self.sums.get(s, 0) + dense.sum(dim=(0, 1))
            elif s in self.read:
                dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=torch.float32)
                self.reads[s] = dense[..., self.read[s]]
        return x


def fwd(inference, tokens, patcher, logits=False):
    inference.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            _, lg, _ = inference.forward(tokens, patcher=patcher, grad_enabled=False, return_activations=False,
                                         tokenize_final=False, all_logits=logits)
    finally:
        inference.enable_compile()
    return lg


def score_target(G, key):
    from circuit.protocol_contexts import eval_probe
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_means
    from eval.wcm_eval import TapCircuitOnly, circuit_alphas, contrast_split, site_encoder
    inference, bank, dev = G["inference"], G["bank"], G["device"]
    K, D = int(bank.k), int(bank.d_sae)
    kinds = list(bank.kinds)
    rec = G["R"].contexts(key)
    probe = eval_probe(rec, "strong", dev)
    n_tr = int(probe.metadata["n_train"])
    pt, pa = probe.pos_tokens[:64], probe.pos_argmax[:64]
    pt_tr, pt_ho, pa_ho = pt[:n_tr], pt[n_tr:], pa[n_tr:]
    nt = probe.neg_tokens[:64]
    nt_tr = nt[:contrast_split(int(nt.shape[0]), probe.metadata)]     # 086 rebuilt contrast: n_neg_train
    l, kd, i = key.split(".")
    layer, sl = int(l), int(i)
    site = (layer, kd)
    W, b = site_encoder(bank, layer, kd)
    w_seed, b_seed = W[sl], b[sl]
    UPS = set(upstream_sites(bank, layer, kd))
    down = [(l2, k2) for l2 in range(bank.n_layer) for k2 in kinds
            if (l2, k2) != site and site in set(upstream_sites(bank, l2, k2))]
    row = dict(seed=key, layer=layer, kind=kd, band=SE.band(layer), n_down_sites=len(down))
    with torch.no_grad():
        e1 = torch.zeros(1, 1, D, device=dev)
        e1[..., sl] = 1.0
        dec = (bank.decode(e1, kd, layer) - bank.decode(torch.zeros_like(e1), kd, layer)).float().reshape(-1)

    # ---- readers, on train contexts
    sums = {}
    for tag, val in (("clean", None), ("abl", "zero")):
        acc = {}
        for s0 in range(0, int(pt_tr.shape[0]), SE.BATCH):
            p = SpliceRead(bank, site, sl, val, down, "sums", dec=dec)
            fwd(inference, pt_tr[s0:s0 + SE.BATCH], p)
            for s, v in p.sums.items():
                acc[s] = acc.get(s, 0) + v
        sums[tag] = acc
    cand = []
    for s in down:
        dlt = sums["clean"][s] - sums["abl"][s]
        top = dlt.abs().topk(min(N_READ, D))
        cand += [(float(dlt[j]), s, int(j)) for j in top.indices.tolist()]
    cand.sort(key=lambda t: -abs(t[0]))
    readers = [c for c in cand[:N_READ] if abs(c[0]) > 1e-6]
    row["n_readers"] = len(readers)
    row["readers"] = [("%d.%s.%d" % (s[0], s[1], j), round(v, 3)) for v, s, j in readers[:10]]
    read = {}
    for _, s, j in readers:
        read.setdefault(s, []).append(j)
    read = {s: torch.tensor(v, device=dev, dtype=torch.long) for s, v in read.items()}
    sign = {s: torch.tensor([1.0 if v > 0 else -1.0 for v, s2, _ in readers if s2 == s], device=dev) for s in read}

    # ---- circuit values for the target on held-out contexts (full [16, T] post-Top-K read)
    means_neg = collect_site_means(inference, bank, nt_tr, UPS) if UPS else {}
    specs = {}
    for name, path in (("wcm", SE.MAIN_DIR / ("%s.pt" % key)), ("rand", SE.RAND_DIR / ("%s.d0.pt" % key))):
        if path.exists():
            c = torch.load(path, weights_only=False)
            if c is not None:
                al = circuit_alphas(c)
                kt = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in al.items() if d}
                ks = {}
                for s, t in kt.items():
                    sv = torch.ones(D, device=dev)
                    sv[t] = torch.tensor([al[s][int(j)] for j in t.tolist()], device=dev)
                    ks[s] = sv
                specs[name] = (kt, ks)
    specs["empty"] = ({}, None)

    def circ_val(kt, ks, means, rtk):
        return SE.run_full(inference, lambda a, b_: TapCircuitOnly(
            bank=bank, keep_indices={s: set(t.tolist()) for s, t in kt.items()}, in_scope=UPS, seed_layer=layer,
            seed_kind=kd, seed_latent_idx=sl, pos_argmax=pa_ho[a:b_], site_means=means, respect_topk=rtk, topk=K,
            keep_tensors=kt, keep_scales=ks, w_seed=w_seed, b_seed=b_seed), pt_ho)

    from eval.wcm_eval import NaturalTap
    nat = SE.run_full(inference, lambda a, b_: NaturalTap(bank, site, w_seed, b_seed, sl), pt_ho)
    values = {"nat": nat}
    if "wcm" in specs:
        values["wcm_C"] = circ_val(*specs["wcm"], means_neg, True)
        values["wcm_Z"] = circ_val(*specs["wcm"], None, False)
    if "rand" in specs:
        values["rand_C"] = circ_val(*specs["rand"], means_neg, True)
    values["empty_C"] = circ_val({}, None, means_neg, True)
    row["val_mean_nat"] = float(nat.mean())
    for k_, v in values.items():
        row["val_corr_%s" % k_] = float(torch.corrcoef(torch.stack([v.flatten(), nat.flatten()]))[0, 1]) \
            if float(v.std()) > 0 and float(nat.std()) > 0 else None

    # ---- splice runs on held-out: readers + logits
    def run(val):
        p = SpliceRead(bank, site, sl, val, list(read), "read", read, dec=dec)
        lg = fwd(inference, pt_ho, p, logits=True).float()
        tot = sum(float((p.reads[s] * sign[s]).sum()) for s in read) if read else 0.0
        return tot, F.log_softmax(lg, dim=-1)

    r_clean, lp_clean = run(None)
    r_abl, lp_abl = run("zero")
    p_clean = lp_clean.exp()

    def kl(lp):
        return float((p_clean * (lp_clean - lp)).sum(-1).mean())

    kl_abl = kl(lp_abl)
    row.update(reader_drop=r_clean - r_abl, kl_ablate=kl_abl)
    den = r_clean - r_abl
    for k_, v in values.items():
        r_s, lp_s = run(v)
        row["reader_rec_%s" % k_] = (r_s - r_abl) / den if abs(den) > 1e-6 else None
        k_s = kl(lp_s)
        row["kl_%s" % k_] = k_s
        row["kl_rec_%s" % k_] = 1 - k_s / kl_abl if kl_abl > 1e-9 else None
    return row


def main():
    OUT75.mkdir(parents=True, exist_ok=True)
    path = OUT75 / "rows.jsonl"
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
    print("075 reader-recovered: %d targets, %d done, N_READ %d" % (len(keys), len(done), N_READ), flush=True)
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
        print("[%s] %4d/%d %-16s %s rr_nat=%s rr_wcm=%s kl_abl=%s  %.1fs  ~%.0f min left" % (
            time.strftime("%H:%M"), n_ + 1, len(keys), key, row.get("error", "ok"),
            None if row.get("reader_rec_nat") is None else round(row["reader_rec_nat"], 3),
            None if row.get("reader_rec_wcm_C") is None else round(row["reader_rec_wcm_C"], 3),
            None if row.get("kl_ablate") is None else "%.2g" % row["kl_ablate"], row["secs"],
            left * (time.time() - t0) / n_new / 60), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    if os.environ.get("REPORT"):
        import report
        report.main()
    else:
        main()
