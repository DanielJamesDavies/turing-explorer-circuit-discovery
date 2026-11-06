"""Arena A (2026-11-05): how many WCM circuit members fire on ONE prompt? Clean forward passes only, CPU.

For each panel circuit (out/panel/wcm_l1e-3 + the two reruns), on K = 8 strongest-train prompts (the fit's strongest
training rows: rec["D_train"] for arm B, else rec["strong"]["train"]; highest record peak first) and the 8 highest-peak
held-out strongest prompts (rec["strong"]["held"]), each truncated after the target's anchor (rec arg), the site model
prepending BOS and hiding it from every site (ARENA_A_BOS=1, the driver's convention), counts:
  n_members        circuit size (members, position-collapsed (layer, kind, latent); the seed excluded)
  mem_any          members with code > 0 at any position <= anchor
  mem_pos          (member, position) pairs with code > 0, positions 0..anchor of the prompt (BOS never a site row)
  mem_anchor       members with code > 0 at the anchor
  all_any/all_pos  the same over every latent of the upstream dictionaries (upstream_sites of the target, real
                   sites only: the universe the members are drawn from)
  read / peak      the target's code at the anchor vs the record's stored peak (sanity check)
z2 = latents 24,576.. of Kissane L5 / L7, a separate site here; ids are distinct, so (site, latent) counts once.
Also reads the Ge / Dunefsky smoke circuit files (per-prompt node lists) for reference.

  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python \
      experiments/107-true-baselines/arena_a/fire_count_tcz.py
Env: STOP_AT (HH:MM, default 11:50): no new circuit starts after it; results so far are written.
"""
import json
import os
import statistics as S
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("ARENA_A_BOS", "1")
os.environ.setdefault("ARENA_A_SCOPE", "a1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wcm_tcz as WT  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", 4)))
A = HERE / "out"
OUT = A / "panel" / "fire_count"
CDIRS = [A / "panel" / "wcm_l1e-3" / "main" / "circuits", A / "panel" / "wcm_l1e-3_rerun" / "main" / "circuits"]
RDIR = A / "panel" / "rec" / "ctx"
K = 8
PASSED = {"2.z.1360", "4.tc.13440", "8.tc.15982"}
STOP_AT = os.environ.get("STOP_AT", "11:50")


def late():
    h, m = map(int, STOP_AT.split(":"))
    now = datetime.now()
    return (now.hour, now.minute) >= (h, m)


def prompts(rec, which):
    s = rec["strong"]
    if which == "train":
        rows = list(rec["D_train"]) if rec.get("mid") is not None and rec.get("D_train") else list(s["train"])
    else:
        rows = list(s["held"])
    rows = sorted(rows, key=lambda j: -float(s["peak"][j]))[:K]
    return [(int(j), s["pos"][j], int(s["arg"][j]), float(s["peak"][j])) for j in rows]


def count_prompt(model, bank, site, latent, ups, members, toks):
    res = dict(mem_any=0, mem_pos=0, mem_anchor=0, all_any=0, all_pos=0, read=None, per_site={})

    def tr(l, k, x):
        if (l, k) == site:
            lat, _ = bank.encode(x, k, l)
            res["read"] = float(lat[0, -1, latent])
            return None
        if (l, k) not in ups:
            return None
        lat, _ = bank.encode(x, k, l)
        act = lat[0] > 0                                           # [T, d_sae], T = anchor + 1, BOS excluded
        res["all_pos"] += int(act.sum())
        res["all_any"] += int(act.any(0).sum())
        m = members.get((l, k))
        if m is not None:
            am = act[:, m]
            ps = dict(n=int(m.numel()), any=int(am.any(0).sum()), pos=int(am.sum()), anchor=int(am[-1].sum()))
            res["mem_any"] += ps["any"]
            res["mem_pos"] += ps["pos"]
            res["mem_anchor"] += ps["anchor"]
            res["per_site"]["%d.%s" % (l, k)] = ps
        return None

    with torch.no_grad(), model.patch_sites(tr):
        model(toks.view(1, -1))
    return res


def med_rng(v):
    return dict(median=S.median(v), min=min(v), max=max(v)) if v else None


def ge_reference():
    """per (target, prompt): Ge hierarchical node counts at tau / a_t ~ 1e-3, 1e-2, 1e-1 ((latent, pos) with pos 0 =
    BOS dropped, and distinct (site, latent)); Dunefsky distinct tc latents and (latent, token) rows per num_branches."""
    out = dict(ge=[], dunefsky=[])
    for p in sorted((A / "ge_smoke_up" / "circuits").glob("*/p*.pt")):
        try:
            d = torch.load(p, weights_only=False, map_location="cpu")
        except Exception as e:                                      # noqa: BLE001
            out["ge"].append(dict(file=str(p), error="%s: %s" % (type(e).__name__, str(e)[:200])))
            continue
        if d.get("status") != "ok":
            continue
        a_t = float(d["a_t"])
        row = dict(key=d["key"], j=d["j"], T=len(d["tokens"]) - 1, a_t=a_t, by_tau={})
        for want in (1e-3, 1e-2, 1e-1):
            h = min(d["hier"], key=lambda h: abs(float(h["tau"]) / a_t - want))
            n = h["nodes"]
            n = n[n[:, 1] > 0] if n.numel() else n
            lat = {(int(a), int(c)) for a, _, c in n.tolist()}
            row["by_tau"]["%g" % want] = dict(tau_rel=float(h["tau"]) / a_t, latent_pos=int(n.shape[0]),
                                              latents=len(lat))
        out["ge"].append(row)
    for p in sorted((A / "dunefsky_smoke" / "circuits").glob("*/p*.pt")):
        try:
            d = torch.load(p, weights_only=False, map_location="cpu")
        except Exception as e:                                      # noqa: BLE001
            out["dunefsky"].append(dict(file=str(p), error="%s: %s" % (type(e).__name__, str(e)[:200])))
            continue
        if d.get("status") != "ok":
            continue
        rows = d.get("rows") or (d.get("graph") or {}).get("rows") or []
        tc_rows = {(r["layer"], r["latent"], r["token"]) for r in rows
                   if isinstance(r, dict) and r.get("type") == "mlp" and r.get("latent") is not None
                   and not r.get("root")}
        out["dunefsky"].append(dict(key=d["key"], j=d["j"], num_branches=d["num_branches"],
                                    n_tc_latents=d.get("n_tc_latents"), n_tc_latent_token=len(tc_rows) or None))
    return out


def main():
    from eval.ablation_faithfulness import upstream_sites
    from eval.wcm_eval import circuit_alphas, circuit_key
    OUT.mkdir(parents=True, exist_ok=True)
    model, bank = WT.load_model(device="cpu", preload=False)
    model.eval()
    real = set(WT.real_sites())
    paths = sorted([p for d in CDIRS for p in d.glob("*.pt")],
                   key=lambda p: (int(p.stem.split(".")[0]), p.stem))
    results, stopped = [], None
    for cp in paths:
        if late():
            stopped = "STOP_AT %s reached before %s" % (STOP_AT, cp.stem)
            print(stopped, flush=True)
            break
        t0 = time.time()
        c = torch.load(cp, weights_only=False, map_location="cpu")
        if c is None:
            continue
        key = circuit_key(c)
        l, k, i = int(key.split(".")[0]), key.split(".")[1], int(key.split(".")[2])
        alphas = circuit_alphas(c)
        ups = {s for s in upstream_sites(bank, l, k) if s in real}
        members = {s: torch.tensor(sorted(d), dtype=torch.long) for s, d in alphas.items() if d}
        outside = sorted("%d.%s" % s for s in members if s not in ups)
        rec = torch.load(RDIR / ("%s.pt" % key), weights_only=False)[key]
        res = dict(key=key, passed=key in PASSED, n_members=sum(len(d) for d in alphas.values()),
                   members_outside_upstream=outside, n_up_sites=len(ups), arm="B" if rec.get("mid") is not None else "A",
                   prompts={})
        for which in ("train", "held"):
            rows = []
            for j, pos, arg, peak in prompts(rec, which):
                toks = pos[:arg + 1].long()
                r = count_prompt(model, bank, (l, k), i, ups, members, toks)
                r.update(row=j, anchor=arg, peak=peak, rel_err=abs(r["read"] - peak) / max(abs(peak), 1e-9))
                rows.append(r)
            res["prompts"][which] = rows
            res[which] = {f: med_rng([r[f] for r in rows]) for f in
                          ("mem_any", "mem_pos", "mem_anchor", "all_any", "all_pos", "anchor")}
            res[which]["max_rel_err"] = max(r["rel_err"] for r in rows)
        res["secs"] = round(time.time() - t0, 1)
        results.append(res)
        print("%-12s n %5d | train mem_any %s pos %s anc %s | held mem_any %s pos %s anc %s | err %.1e | %.0fs" % (
            key, res["n_members"], res["train"]["mem_any"]["median"], res["train"]["mem_pos"]["median"],
            res["train"]["mem_anchor"]["median"], res["held"]["mem_any"]["median"], res["held"]["mem_pos"]["median"],
            res["held"]["mem_anchor"]["median"], max(res["train"]["max_rel_err"], res["held"]["max_rel_err"]),
            res["secs"]), flush=True)
        json.dump(dict(results=results, stopped=stopped), open(OUT / "fire_count.json", "w"), indent=1)
    ref = None
    if not late() and os.environ.get("REF", "1") == "1":
        try:
            ref = ge_reference()
        except Exception as e:                                      # noqa: BLE001
            ref = dict(error="%s: %s" % (type(e).__name__, str(e)[:300]))
    json.dump(dict(results=results, stopped=stopped, reference=ref), open(OUT / "fire_count.json", "w"), indent=1)
    print("fire_count: done (%d circuits)%s" % (len(results), "" if stopped is None else "; " + stopped), flush=True)


if __name__ == "__main__":
    main()
