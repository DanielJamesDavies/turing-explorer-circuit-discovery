"""D4.3 (containment) + D4.4 (anatomy) + D3.4 (signed-role pairing) —
pure-CPU passes over the archives. No GPU, no rediscovery.

D4.3  Are drivers a SUBSET of closure membership? Overlap of driver sets
      (AMPC direct-mass K, restoration heads) with the abl-mask closure
      circuit, split by role, against the base rate.
D4.4  Driver ANATOMY: layer distance from seed, near-seed fraction,
      direct-vs-mediated share, per-definition, across the panel.
D3.4  Signed-role pairing for masks: stamp restoration's attribution
      SIGNS onto the abl-mask (role-blind) members, giving the closure
      circuit a role composition it cannot produce itself.

  python experiments/014-d4-analyses/cpu_analyses.py
"""
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import torch

HERE = Path(__file__).parent
ROOT = HERE.parent
D1 = ROOT / "012-driver-bakeoff"
D22 = ROOT / "019-roles-drivers"
D36 = ROOT / "018-maskrefine"
INH = ROOT / "022-inhibmask"
D_SAE = 40960
KS = (16, 64, 256, 1024)

PANEL = [(2, 19766, 0, "resid", 2), (8, 20333, 2, "resid", 8),
         (9, 38734, 3, "attn", 9), (13, 30053, 4, "mlp", 13),
         (17, 38268, 5, "resid", 17), (20, 35678, 6, "resid", 20),
         (25, 10628, 8, "mlp", 25), (26, 17432, 8, "resid", 26),
         (27, 6859, 9, "attn", 27), (29, 2753, 9, "resid", 29),
         (35, 6599, 11, "resid", 35)]


def load_rank(path, limit=None):
    out = []
    if not path.exists():
        return out
    with gzip.open(path, "rt", encoding="utf-8") as gz:
        for i, line in enumerate(gz):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            if len(rec) == 6:
                s, l, kd, idx, role, rr = rec
                out.append(((l, kd), int(idx), role))
            else:
                l, kd, idx, v = rec
                out.append(((l, kd), int(idx), None))
    return out


def direct_rank(sc, sl, k=4096):
    p = D1 / ("direct_full_%d_%d.pt" % (sc, sl))
    if not p.exists():
        return []
    dw = torch.load(p, map_location="cpu", weights_only=False)["direct"]
    tri = []
    for s, w in dw.items():
        v, ix = torch.topk(w, k=min(k, w.numel()))
        tri += [(float(vv), s, int(ii)) for vv, ii in zip(v, ix)]
    tri.sort(key=lambda x: -x[0])
    return [(s, i) for _, s, i in tri]


rows43, rows44, rows34 = [], [], []
for sc, sl, layer, kind, n_sites in PANEL:
    seed = "%d/%d" % (sc, sl)
    R = load_rank(D22 / ("ranking_R_%d_%d.jsonl.gz" % (sc, sl)), limit=200000)
    MF = load_rank(D36 / ("members_MF_%d_%d.jsonl.gz" % (sc, sl)))
    C = direct_rank(sc, sl)
    if not (R and MF and C):
        print("skip %s (missing archives)" % seed)
        continue
    mf_set = {(s, i) for s, i, _ in MF}
    r_sign = {(s, i): role for s, i, role in R}
    base = len(mf_set) / (n_sites * D_SAE)

    # ---- D4.3 containment ------------------------------------------
    for name, rank in (("C_direct", [(s, i) for s, i in C]),
                       ("R_head", [(s, i) for s, i, _ in R])):
        for K in KS:
            head = rank[:K]
            if not head:
                continue
            inside = sum(1 for m in head if m in mf_set)
            rows43.append({"seed": seed, "layer": layer, "kind": kind,
                           "driver": name, "K": K,
                           "in_closure": round(inside / len(head), 4),
                           "base_rate": round(base, 5),
                           "enrichment": round((inside / len(head)) / base, 1)
                           if base else None})

    # ---- D4.4 anatomy ----------------------------------------------
    direct_top = {(s, i) for s, i in C[:1024]}
    for name, rank in (("C_direct", [(s, i) for s, i in C]),
                       ("R_head", [(s, i) for s, i, _ in R])):
        for K in (64, 1024):
            head = rank[:K]
            if not head:
                continue
            dist = [layer - s[0] for s, _ in head]
            rows44.append({
                "seed": seed, "layer": layer, "kind": kind, "driver": name,
                "K": K,
                "mean_dist": round(sum(dist) / len(dist), 2),
                "near2_frac": round(sum(1 for d in dist if d <= 2) / len(dist), 3),
                "same_layer_frac": round(sum(1 for d in dist if d == 0) / len(dist), 3),
                "direct_frac": round(sum(1 for m in head if m in direct_top)
                                     / len(head), 3),
                "kind_mix": {k: round(sum(1 for s, _ in head if s[1] == k)
                                      / len(head), 3)
                             for k in ("attn", "mlp", "resid")}})

    # ---- D3.4 signed-role pairing for the closure mask --------------
    labelled = [(m, r_sign.get(m)) for m in mf_set]
    n_lab = sum(1 for _, r in labelled if r)
    n_inh = sum(1 for _, r in labelled if r == "counterfactual_inhibitor")
    n_act = n_lab - n_inh
    # brake set (learned, intervention-native) for comparison
    br = INH / ("members_v2_lam0.001_%d_%d.jsonl.gz" % (sc, sl))
    brakes = {(s, i) for s, i, _ in load_rank(br)} if br.exists() else set()
    rows34.append({"seed": seed, "layer": layer, "kind": kind,
                   "n_closure": len(mf_set),
                   "labelled_frac": round(n_lab / len(mf_set), 4),
                   "inhib_frac_of_labelled": round(n_inh / max(n_lab, 1), 4),
                   "n_inh": n_inh, "n_act": n_act,
                   "n_learned_brakes_inside": len(brakes & mf_set) if brakes else None,
                   "learned_brakes_labelled_inhib": (
                       round(sum(1 for m in (brakes & mf_set)
                                 if r_sign.get(m) == "counterfactual_inhibitor")
                             / max(len(brakes & mf_set), 1), 4)
                       if brakes else None)})
    print("done %s" % seed, flush=True)

for name, rows in (("d43_containment", rows43), ("d44_anatomy", rows44),
                   ("d34_signed_roles", rows34)):
    with (HERE / ("%s.jsonl" % name)).open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

# ---------------- summaries ----------------
out = ["# D4.3 / D4.4 / D3.4 — CPU analyses (2026-08-02)\n"]
out.append("## D4.3 — Containment: are drivers inside the closure?\n")
out.append("| driver set | K | median in-closure | median base rate | median enrichment |")
out.append("|---|---:|---:|---:|---:|")
for name in ("C_direct", "R_head"):
    for K in KS:
        sel = [r for r in rows43 if r["driver"] == name and r["K"] == K]
        if not sel:
            continue
        out.append("| %s | %d | %.0f%% | %.2f%% | %.0fx |" % (
            name, K, 100 * median(r["in_closure"] for r in sel),
            100 * median(r["base_rate"] for r in sel),
            median(r["enrichment"] for r in sel)))
out.append("\nPer-seed at K=64 (C_direct):\n")
out.append("| seed | L/kind | in closure | enrichment |")
out.append("|---|---|---:|---:|")
for r in [x for x in rows43 if x["driver"] == "C_direct" and x["K"] == 64]:
    out.append("| %s | L%d %s | %.0f%% | %.0fx |" % (
        r["seed"], r["layer"], r["kind"], 100 * r["in_closure"], r["enrichment"]))

out.append("\n## D4.4 — Driver anatomy\n")
out.append("| driver | K | median dist | median near<=2 | same-layer | direct-top1k | kind mix (a/m/r) |")
out.append("|---|---:|---:|---:|---:|---:|---|")
for name in ("C_direct", "R_head"):
    for K in (64, 1024):
        sel = [r for r in rows44 if r["driver"] == name and r["K"] == K]
        if not sel:
            continue
        out.append("| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f/%.2f/%.2f |" % (
            name, K, median(r["mean_dist"] for r in sel),
            median(r["near2_frac"] for r in sel),
            median(r["same_layer_frac"] for r in sel),
            median(r["direct_frac"] for r in sel),
            median(r["kind_mix"]["attn"] for r in sel),
            median(r["kind_mix"]["mlp"] for r in sel),
            median(r["kind_mix"]["resid"] for r in sel)))

out.append("\n## D3.4 — Signed-role pairing for the closure mask\n")
out.append("| seed | L/kind | closure n | labelled by R | inhibitor share | learned brakes inside | of those, R-labelled inhib |")
out.append("|---|---|---:|---:|---:|---:|---:|")
for r in rows34:
    out.append("| %s | L%d %s | %d | %.0f%% | %.0f%% | %s | %s |" % (
        r["seed"], r["layer"], r["kind"], r["n_closure"],
        100 * r["labelled_frac"], 100 * r["inhib_frac_of_labelled"],
        r["n_learned_brakes_inside"],
        ("%.0f%%" % (100 * r["learned_brakes_labelled_inhib"]))
        if r["learned_brakes_labelled_inhib"] is not None else "-"))
out.append("\nmedian inhibitor share of labelled closure members: %.0f%%"
           % (100 * median(r["inhib_frac_of_labelled"] for r in rows34)))
(HERE / "cpu_analyses_summary.md").write_text("\n".join(out) + "\n",
                                              encoding="utf-8")
print("\n".join(out))
