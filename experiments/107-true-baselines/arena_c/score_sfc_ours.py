"""Arena C (DAN-196) cross-scoring (a): SFC's circuits under OUR tests.

Input: SFC's nodes-only per-position circuits, out/sfc/circuits/pythia-70m-deduped_<task>_train_n100_aggnone_nodeall.pt
(nodes[name] = SparseAct(act [T, 32768], resc [T, 1]): the mean IG indirect effect per (position, latent) and per
(position, error)), at SFC's 15 node thresholds (torch.logspace(-4, 0, 15), their notebook's grid); node = |IE| > T_N.

Conversion [ours, stated in WCM_ADAPTER.md / README]:
  - members = the position-collapsed union of the feature nodes at our 18 noembed sites (a latent is a member if
    |IE| > T_N at ANY position), every member at alpha = 1 (SFC has no coefficients);
  - embed nodes are dropped (the embedding is out of WCM's scope on this substrate);
  - error nodes are not used: our scorer always keeps every SAE error at its clean value;
  - SFC's native counts are recorded beside ours: feature nodes (position, latent) and error nodes (position, site),
    over all sites and at layers >= 2 (what SFC's evaluator counts), and the distinct latents.
Scoring: wcm_marks' engine.score (Z / A / C / necessity; means pooled AND per_position) and donor_tasks (recovery A /
C / R, fit = a1 here since alpha = 1), on SFC's 40 held-out prompts (data mode "sfc"), exactly as the WCM rows.
Output: out/cross/sfc_ours.jsonl, one row per (task, threshold, means); resumable; STOP_EPOCH / STOP_AT respected.

  HF_HUB_OFFLINE=1 ./.venv/bin/python experiments/107-true-baselines/arena_c/score_sfc_ours.py [--tasks ...]
"""
import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wcm_marks as WM  # noqa: E402
import donor_tasks as DN  # noqa: E402
import torch  # noqa: E402

SFC_DIR = HERE / "out" / "sfc" / "circuits"
OUT = HERE / "out" / "cross"
THRESHOLDS = torch.logspace(-4, 0, 15).tolist()      # their notebook cell 5


def load_sfc_nodes(task: str):
    """{sfc submodule name: (act [T, D] float, resc [T, 1] float)} from SFC's nodes-only circuit file, via the plain-
    tensor export (score_wcm_sfc.py --export, in the SFC env: their pickle imports nnsight)."""
    p = OUT / ("sfc_nodes_%s.pt" % task)
    d = torch.load(p, map_location="cpu", weights_only=True)
    return {k: (v["act"], v["resc"]) for k, v in d["nodes"].items()}, HERE / d["source"]


def site_of(name: str):
    if name == "embed":
        return (0, "embed")
    kind, layer = name.rsplit("_", 1)
    return (int(layer), kind)


def convert(nodes, threshold: float, sites):
    """(alphas {site: {latent: 1.0}} at our scope, native counts)."""
    sites = set(sites)
    alphas, nat = {}, dict(feat_all=0, err_all=0, feat_ge2=0, err_ge2=0, distinct_all=0, distinct_ge2=0,
                           feat_embed=0)
    for name, (act, resc) in nodes.items():
        s = site_of(name)
        fa = act.abs() > threshold
        nf, ne = int(fa.sum()), int((resc.abs() > threshold).sum())
        nd = int(fa.any(0).sum())
        nat["feat_all"] += nf
        nat["err_all"] += ne
        nat["distinct_all"] += nd
        if s[1] == "embed":
            nat["feat_embed"] += nf
        elif s[0] >= 2:
            nat["feat_ge2"] += nf
            nat["err_ge2"] += ne
            nat["distinct_ge2"] += nd
        if s in sites:
            idx = fa.any(0).nonzero().flatten().tolist()
            if idx:
                alphas[s] = {int(i): 1.0 for i in idx}
    nat["sfc_nodes_ge2"] = nat["feat_ge2"] + nat["err_ge2"]          # = SFC evaluator's n_nodes (start_layer 2)
    return alphas, nat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(WM.SFC_TASKS))
    ap.add_argument("--sites", default="noembed")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sfc_ours.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r["task"], round(r["threshold"], 9), r["means"]))
            except Exception:  # noqa: BLE001
                pass
    started = _dt.datetime.now()
    sites = WM.scope_sites(a.sites)
    eng = {}
    for task in a.tasks.split(","):
        todo = [(th, m) for th in THRESHOLDS for m in WM.MEANS_MODES if (task, round(th, 9), m) not in done]
        if not todo:
            continue
        if WM.stop_passed(os.environ.get("STOP_AT"), started):
            print("deadline passed: stopping", flush=True)
            return
        if not eng:
            eng["pooled"] = WM.make_engine(a.sites, means="pooled")
            eng["per_position"] = WM.make_engine(a.sites, means="per_position",
                                                 shared=(eng["pooled"].inf, eng["pooled"].bank))
        nodes, src = load_sfc_nodes(task)
        T = WM.build_task(task, "sfc")
        Bs = {m: eng[m].batch(T) for m in WM.MEANS_MODES}
        plen = T["prompt_len_train"][0]
        conv = {th: convert(nodes, th, sites) for th in THRESHOLDS}
        # donor test once per threshold (independent of the means mode), all thresholds under the same donors
        ts = time.time()
        ctx = DN.task_context(task, "sfc", eng["pooled"].inf, eng["pooled"].bank)
        ths = sorted({th for th, _ in todo})
        dres = DN.score_group(eng["pooled"].inf, eng["pooled"].bank, ctx, sites,
                              [dict(alphas=conv[th][0]) for th in ths])
        donor = dict(zip(ths, dres))
        t_donor = time.time() - ts
        for th, m in todo:
            alphas, nat = conv[th]
            ts = time.time()
            sc = eng[m].score(Bs[m], alphas) if alphas else None
            dr = donor[th]
            row = dict(method="SFC", task=task, sfc_task=task, threshold=th, means=m, sites=a.sites, data_mode="sfc",
                       conversion="position-collapsed union of feature nodes at the scope's sites, alpha = 1, embed "
                                  "and error nodes dropped (ours)", circuit=str(src.relative_to(HERE)),
                       native=nat, **WM.sizes(alphas, plen), prompt_len=plen, t_score=round(time.time() - ts, 2),
                       t_donor_task=round(t_donor, 1),
                       recovery=dr["recovery"], empty_leak=dr["empty_leak"])
            if sc is not None:
                row.update({k: v for k, v in sc.items() if not isinstance(v, (dict, list))})
            from eval.wcm_eval import passes_task
            row["prod_pass"] = bool(sc is not None and not row.get("task_vacuous") and passes_task(row))
            row["resample_pass"] = DN.resample_pass(row if sc is not None else None,
                                                    {ds: dr["recovery"][ds]["fit"] for ds in DN.POOLS})
            with open(path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print("%s T %.3g %s: %d latents (SFC nodes>=2 %d) | Z %s A %s C %s nec %s | donor A %s C %s R %s | pass %s"
                  % (task, th, m, row["n_latents"], nat["sfc_nodes_ge2"], *(DN._f(row.get("task_" + k)) for k in
                     ("free0", "freeM_topk", "freeN_topk", "nec")),
                     *(DN._f(dr["recovery"][d]["fit"]) for d in DN.POOLS), row["prod_pass"]), flush=True)


if __name__ == "__main__":
    main()
