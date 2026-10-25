"""073 diagnostics: why do circuits fail under resample ablation?

Per target (stratified sample, PER_CELL per band x kind), donors R (random corpus) and C (contrast train), N_DRAWS:
  read      post-Top-K (tk, the pass-rule read) AND relu(w.x + b) (pre, uncensored). tk << pre means the target's
            pre-activation is reproduced but it is pushed out of its site's Top-K by the donor's competitors.
  members   live (re-encoded from the mixed stream, as scored) vs PINNED to their own natural values in the base
            context, position by position (alpha applied). pinned >> live means the node set carries the signal but
            the members' own inputs from outside the circuit are lost (closure); pinned also failing means the donor
            background overrides the target even with the members exactly right (competition).
  dose      a random fraction f of the donor's non-member entries kept, the rest zero: f = 0 is the Z fill,
            f = 1 is the full resample.
  self      the base context as its own donor: must give ~1 (identity, up to bf16 round trip).
Score = (x - e) / (a_pos - e) at the held-out anchors, per read, e = the empty circuit under the same fill.

  PYTHONPATH=src python experiments/073-strict-eval/diag.py      env: PER_CELL (5)  N_DRAWS (2)  -> out/diag.jsonl
"""
import json
import os
import sys
import time
import traceback
import zlib
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import strict_eval as SE  # noqa: E402

PER_CELL = int(os.environ.get("PER_CELL", 5))
N_DRAWS = int(os.environ.get("N_DRAWS", 2))
FRACS = (0.25, 0.5, 1.0)
OUT = SE.OUT73 / "diag.jsonl"


def pick(keys):
    from collections import defaultdict
    cells = defaultdict(list)
    for k in keys:
        l, kd, _ = k.split(".")
        cells[(SE.band(int(l)), kd)].append(k)
    return [k for c in sorted(cells) for k in cells[c][:PER_CELL]]


def score_target(G, key, pool):
    from circuit.protocol_contexts import eval_probe
    from eval.ablation_faithfulness import upstream_sites
    from eval.wcm_eval import NaturalTap, circuit_alphas, contrast_split, site_encoder
    inference, bank, dev = G["inference"], G["bank"], G["device"]
    K = int(bank.k)
    rec = G["R"].contexts(key)
    probe = eval_probe(rec, "strong", dev)
    n_tr = int(probe.metadata["n_train"])
    pt, pa = probe.pos_tokens[:64], probe.pos_argmax[:64]
    pt_ho, pa_ho = pt[n_tr:], pa[n_tr:]
    nt = probe.neg_tokens[:64]
    nt_tr = nt[:contrast_split(int(nt.shape[0]), probe.metadata)]     # 086 rebuilt contrast: n_neg_train
    l, kd, i = key.split(".")
    layer, sl = int(l), int(i)
    site = (layer, kd)
    W, b = site_encoder(bank, layer, kd)
    w_seed, b_seed = W[sl], b[sl]
    UPS = set(upstream_sites(bank, layer, kd))
    sites = sorted(UPS)
    c = torch.load(SE.MAIN_DIR / ("%s.pt" % key), weights_only=False)
    al = circuit_alphas(c)
    kt = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in al.items() if d}
    sc = {s: torch.tensor([al[s][int(j)] for j in kt[s].tolist()], device=dev) for s in kt}
    n_ho = int(pt_ho.shape[0])

    nat_tk, nat_pre = SE.run_full(inference, lambda a, b_: NaturalTap(bank, site, w_seed, b_seed, sl), pt_ho, pre=True)
    a_pos = {"tk": float(SE.at(nat_tk, pa_ho).mean()), "pre": float(SE.at(nat_pre, pa_ho).mean())}
    base_codes = SE.capture_codes(inference, bank, pt_ho, sites)
    # pins: each member's natural value in the base context, every position [n_ho, T, n_members_at_site]
    pins = {}
    for s, t in kt.items():
        ta, ti = base_codes[s]
        dense = torch.zeros(ta.shape[0], ta.shape[1], int(bank.d_sae), device=ta.device)
        dense.scatter_(-1, ti, ta)
        pins[s] = dense[..., t].clone()
        del dense
    row = dict(seed=key, layer=layer, kind=kd, band=SE.band(layer), n=sum(int(t.numel()) for t in kt.values()),
               a_pos_tk=a_pos["tk"], a_pos_pre=a_pos["pre"])

    def go(donor, kt_, sc_, frac=1.0, pins_=None, gs=0, mpre=False):
        def make(a, b_):
            p = SE.ResamplePatcher(
                bank, UPS, site, w_seed, b_seed, sl, kt_, sc_, {s: (x[a:b_], y[a:b_]) for s, (x, y) in donor.items()},
                K, frac=frac, pins=None if pins_ is None else {s: v[a:b_] for s, v in pins_.items()}, gen_seed=gs)
            p.member_pre = mpre
            return p
        tk, pre = SE.run_full(inference, make, pt_ho, pre=True)
        return {"tk": float(SE.at(tk, pa_ho).mean()), "pre": float(SE.at(pre, pa_ho).mean())}

    def ratio_of(xs, es, rd):
        x = sum(v[rd] for v in xs) / len(xs)
        e = sum(v[rd] for v in es) / len(es)
        return SE.ratio(x, e, a_pos[rd])

    # self-donor identity
    s_e = go(base_codes, {}, {})
    s_c = go(base_codes, kt, sc)
    row["self_empty_rel"] = {rd: s_e[rd] / a_pos[rd] if a_pos[rd] > 1e-9 else None for rd in ("tk", "pre")}
    row["self_wcm_rel"] = {rd: s_c[rd] / a_pos[rd] if a_pos[rd] > 1e-9 else None for rd in ("tk", "pre")}

    g = torch.Generator().manual_seed(zlib.crc32(key.encode()))
    donors = {"C": nt_tr, "R": pool[torch.randperm(pool.shape[0], generator=g)[:n_ho * N_DRAWS]]}
    for ds, toks in donors.items():
        codes = SE.capture_codes(inference, bank, toks, sites)
        res = {}
        for d in range(N_DRAWS):
            sel = (torch.arange(d * n_ho, (d + 1) * n_ho) if ds == "R"
                   else torch.randperm(int(toks.shape[0]), generator=g)[:n_ho])
            donor = {s: (codes[s][0][sel.to(codes[s][0].device)], codes[s][1][sel.to(codes[s][1].device)])
                     for s in sites}
            for f in FRACS:
                res.setdefault(("empty", f), []).append(go(donor, {}, {}, frac=f, gs=d))
                res.setdefault(("live", f), []).append(go(donor, kt, sc, frac=f, gs=d))
            res.setdefault(("pinned", 1.0), []).append(go(donor, kt, sc, pins_=pins, gs=d))
            res.setdefault(("mpre", 1.0), []).append(go(donor, kt, sc, gs=d, mpre=True))
            res.setdefault(("mpre_a1", 1.0), []).append(go(donor, kt, {}, gs=d, mpre=True))
        for (v, f), xs in res.items():
            if v == "empty":
                continue
            for rd in ("tk", "pre"):
                row["%s_%s_f%g_%s" % (ds, v, f, rd)] = ratio_of(xs, res[("empty", f)], rd)
        for rd in ("tk", "pre"):
            row["%s_leak_%s" % (ds, rd)] = (sum(v[rd] for v in res[("empty", 1.0)]) / N_DRAWS) / a_pos[rd] \
                if a_pos[rd] > 1e-9 else None
        del codes
    return row


def main():
    keys = pick(SE.order_targets(SE.default_targets()))
    R = SE.driver.Runner()
    G = dict(R.G)
    G["R"] = R
    pool = torch.load(SE.OUT73 / "random_pool.pt")["tokens"].to(G["device"])
    done = set()
    if OUT.exists():
        done = {json.loads(l)["seed"] for l in open(OUT) if "error" not in json.loads(l)}
    print("073 diag: %d targets, %d done" % (len(keys), len(done)), flush=True)
    fh = open(OUT, "a")
    for n_, key in enumerate(keys):
        if key in done:
            continue
        ts = time.time()
        try:
            row = score_target(G, key, pool)
        except Exception as e:  # noqa: BLE001
            row = dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))
            traceback.print_exc()
        row["secs"] = round(time.time() - ts, 1)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("[%s] %3d/%d %-16s %s  C live tk %s pre %s | pinned tk %s pre %s | self %s  %.0fs" % (
            time.strftime("%H:%M"), n_ + 1, len(keys), key, row.get("error", "ok"),
            *["%.2f" % row[k] if row.get(k) is not None else "-" for k in
              ("C_live_f1_tk", "C_live_f1_pre", "C_pinned_f1_tk", "C_pinned_f1_pre")],
            "%.2f" % row["self_wcm_rel"]["tk"] if row.get("self_wcm_rel") and row["self_wcm_rel"]["tk"] else "-",
            row["secs"]), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
