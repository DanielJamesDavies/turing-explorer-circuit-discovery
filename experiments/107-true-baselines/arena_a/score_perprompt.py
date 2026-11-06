"""Arena A (A8b, DAN-188, Daniel approved 2026-11-05): the per-prompt ("their turf") test. Does the part of a WCM circuit
that is active on ONE prompt do as well, on that prompt, as a circuit built for that prompt (Ge hierarchical / post-hoc,
Dunefsky)? score_arena_a.py already does the other direction (their circuits on held-out contexts). This is a sibling
script: it imports score_arena_a (its Scorer, units and helpers) and changes nothing in it or in src/.

Units, per (target, WCM lambda arm, prompt) [ours]:
  wcm_full      the whole circuit, fitted alphas, scored on this one prompt (= score_arena_a's home scope on prompt q;
                --check asserts it)
  wcm_fire      the members that FIRE on the prompt: clean run (nothing edited), code > 0 at some position <= the
                anchor, BOS excluded (the site model hides it: ARENA_A_BOS=1), exactly fire_count_tcz.py's definition;
                (latent, pos) = the positions they fire at. Alphas unchanged.
  wcm_fire_top  "ours (adaptation)": the firing members ordered by alpha x (max clean code over the prompt's positions
                <= anchor), ties by (layer, kind, latent); top-k on the grid k = 1, 2, 4, ... < n_fire. Only in the
                --topk-arms arms (default: the first --wcm label), the secondary WCM curve
Tests on that prompt = score_arena_a's per-prompt home evaluation, unchanged (Scorer.unit on a one-row eval set):
  Z / A / C (free0 / freeM_topk / freeN_topk, position-collapsed keep, the same src patchers and fills), nec_a (errors
  live), nec_b (ALL upstream errors held at clean: the Ge / Dunefsky / circuit-tracer convention, the PRIMARY comparison
  column), nec_b_tc (tc error only, diagnostic); a_pos / e0 / eM / eN of that prompt row.
  The keep is position-collapsed (a member is kept at every position), as for the baselines in score_arena_a; the
  (latent, pos) count is recorded. Ge recovery: the unit's nodes at their firing positions (Ge's space, BOS-prefixed
  pos + 1) as ge_recovery_tcz.py jobs (OUT/ge_jobs.pp.jsonl, written while scoring; wcm_full: every non-BOS position,
  as score_arena_a); run ge_recovery_tcz.py (ge107) on them, --summary merges OUT/ge_recovery*.jsonl.
  Dunefsky coverage is not defined for a WCM unit (it is read from the error nodes of THEIR graph search); the baseline
  rows carry theirs + ours (score_arena_a).
Prompt sets:
  train  the K = 8 prompts the baselines get (score_arena_a.prompts_for = make_targets: the K highest-peak strong
         contexts of the WCM training arm). Row q = score_arena_a's home row q. NOTE: WCM was fitted on these.
  held   the K highest-peak rows (record peak, ties by row) of the scorer's held-out strongest eval set ("ho" = the
         first 16 rec["strong"]["held"] rows), each read at its anchor. WCM never saw them. --emit-heldout-prompts DIR
         writes DIR/prompts.jsonl in make_targets' schema, which run_ge.py / run_dunefsky.py read through --targets DIR
         unchanged; --ge-held / --dunefsky-held score those circuits here on the same held rows (prompt asserted).
Baseline points on the train prompts come from score_arena_a's rows.jsonl (--baseline-rows; eval_set home, row j).
Outputs (OUT): rows.jsonl (one row per unit; append + flush; resumable by uid; a torn last line is skipped),
ge_jobs.pp.jsonl, check.jsonl (--check), summary.md / summary.json / curves.jsonl / curves.csv (--summary, CPU, no
model; atomic tmp + replace).

  wsl, repo root:
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= ARENA_A_SCOPE=a1 .venv/bin/python -u \
      experiments/107-true-baselines/arena_a/score_perprompt.py --out <OUT> --wcm l1e-3=out/panel/wcm_l1e-3 \
      [--keys k1,k2] [--sets train,held] [--n-prompts 8] [--check] [--threads 4] [--dry-run] [--stop-epoch T]
  ... score_perprompt.py --summary --out <OUT> [--baseline-rows out/score_smoke/rows.jsonl]
  ... score_perprompt.py --emit-heldout-prompts <DIR> --keys k1,k2      (records only, no model)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import statistics as stats
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_arena_a as SA  # noqa: E402  (stdlib-only at import; the model side loads lazily)

OUT_DEFAULT = HERE / "out" / "perprompt_smoke"
K_PROMPTS = SA.K_PROMPTS
SETS = ("train", "held")
METRICS = ("Z", "A", "C", "nec_a", "nec_b", "nec_b_tc")


# ------------------------------------------------------------------------------------------------ restriction (pure)
def firing_latpos(codes, anchor, bos_row=False):
    """clean codes {(layer, kind): (latents [m], acts [T, m])} -> {(layer, kind, pos, latent)} with code > 0 at a
    position <= anchor. Positions are body positions (0 = the first real token: what the site model shows the sites,
    BOS hidden). bos_row=True: the acts carry a BOS row at index 0 (a BOS-prefixed capture); it is dropped and the rest
    shifted by -1 first, so the result is the same."""
    out = set()
    for (l, k), (lats, acts) in codes.items():
        a = acts[1:] if bos_row else acts
        a = a[: int(anchor) + 1]
        nz = (a > 0).nonzero().tolist() if hasattr(a, "nonzero") else [
            [p, c] for p, row in enumerate(a) for c, v in enumerate(row) if v > 0]
        for p, c in nz:
            out.add((int(l), k, int(p), int(lats[c])))
    return out


def act_max(codes, anchor, bos_row=False):
    """{(layer, kind, latent): max clean code over positions <= anchor} (BOS row dropped as firing_latpos)"""
    out = {}
    for (l, k), (lats, acts) in codes.items():
        a = acts[1:] if bos_row else acts
        a = a[: int(anchor) + 1]
        if hasattr(a, "amax"):
            mx = a.amax(0).tolist() if a.shape[0] else [0.0] * len(lats)
        else:
            mx = [max([row[c] for row in a] or [0.0]) for c in range(len(lats))]
        for c, v in enumerate(mx):
            out[(int(l), k, int(lats[c]))] = float(v)
    return out


def all_latpos(alphas, n_pos):
    """every member at every body position 0..n_pos-1 (the keep-all restriction for the exactness check)"""
    return {(l, k, p, int(i)) for (l, k), d in alphas.items() for i in d for p in range(int(n_pos))}


def restrict(alphas, latpos):
    """the circuit restricted to the latents of `latpos`: ({site: {latent: alpha}}, {site: set}); alphas unchanged.
    Every latpos latent must be a member."""
    keep = {}
    for l, k, _p, i in latpos:
        if int(i) not in alphas.get((l, k), {}):
            raise ValueError("(%d, %s, %d) is not a member" % (l, k, i))
        keep.setdefault((l, k), set()).add(int(i))
    al = {s: {i: float(alphas[s][i]) for i in sorted(v)} for s, v in keep.items()}
    return al, {s: set(v) for s, v in keep.items()}


def topk_order(alphas, amax, latents):
    """the firing members (layer, kind, latent) ordered by alpha x clean max code, descending; ties by the key"""
    return sorted(latents, key=lambda t: (-(float(alphas[(t[0], t[1])][t[2]]) * float(amax.get(t, 0.0))), t))


def k_grid(n):
    """log grid 1, 2, 4, ... strictly below n (n itself is the wcm_fire unit)"""
    out, k = [], 1
    while k < n:
        out.append(k)
        k *= 2
    return out


def latpos_of(latpos, latents):
    s = set(latents)
    return {t for t in latpos if (t[0], t[1], t[3]) in s}


def n_latent(members):
    return sum(len(v) for v in members.values())


def ge_nodes(latpos):
    """(layer, kind, body pos, latent) -> sorted [hook_point, BOS-prefixed pos, dictionary latent] (Ge's space)"""
    out = set()
    for l, k, p, i in latpos:
        hp, li = SA.from_ours(l, k, i)
        out.add((hp, int(p) + 1, li))
    return sorted(out)


def held_order(peaks, n_ho, k=K_PROMPTS):
    """the K held-out eval rows h < n_ho with the highest record peak; ties by h"""
    return sorted(range(int(n_ho)), key=lambda h: (-float(peaks[h]), h))[:k]


def sig(members, alphas):
    return json.dumps(sorted((str(s), sorted((int(i), float(alphas[s][i])) for i in v)) for s, v in members.items() if v))


# ------------------------------------------------------------------------------------------------ prompts (records only)
def held_prompts(rec, k=K_PROMPTS):
    """the held prompt set from the record: eval_probe(rec, "strong") rows n_tr.. (= rec["strong"]["held"][h]), the K
    highest-peak, as dicts (h = index in the ho set, j = row in rec["strong"], anchor, peak, tokens up to the anchor)"""
    from circuit.protocol_contexts import eval_probe
    from eval.wcm_eval import eval_split
    probe = eval_probe(rec, "strong", "cpu")
    pt = probe.pos_tokens[:64]
    n_tr = eval_split(int(pt.shape[0]), probe.metadata, "explicit")
    s = rec["strong"]
    held = list(s["held"])
    n_ho = int(pt.shape[0]) - n_tr
    out = []
    for h in held_order([s["peak"][held[x]] for x in range(n_ho)], n_ho, k):
        j = int(held[h])
        a = int(s["arg"][j])
        assert bool((pt[n_tr + h].cpu() == s["pos"][j].cpu()).all()), ("ho row mismatch", h, j)
        out.append(dict(pool="strong_held", h=int(h), j=j, seq_id=None if s.get("ids") is None else int(s["ids"][j]),
                        anchor=a, peak=float(s["peak"][j]), tokens=[int(t) for t in s["pos"][j][: a + 1]]))
    return out, n_ho


def emit_heldout_prompts(out_dir, keys, k):
    """DIR/prompts.jsonl in make_targets' row schema with the held prompts (run_ge.py / run_dunefsky.py --targets DIR)"""
    import wcm_tcz as WT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in keys:
        rec, path = SA.record_of(key)
        l, kind, i = SA.parse_key(key)
        dk, dl = WT.orig_latent(kind, i)
        pr, n_ho = held_prompts(rec, k)
        from circuit.protocol_contexts import training_arm
        rows.append(dict(key=key, layer=l, kind=kind, latent=i, dict_kind=dk, dict_latent=dl,
                         arm=training_arm(rec)[0], prompt_set="held", n_ho=n_ho, record=str(path), prompts=pr))
    tmp = out_dir / "prompts.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    tmp.replace(out_dir / "prompts.jsonl")
    print("held-out prompts: %d targets x <= %d prompts -> %s" % (len(rows), k, out_dir / "prompts.jsonl"))


# ------------------------------------------------------------------------------------------------ model side
class PerPrompt:
    """score_arena_a.Scorer + the clean capture and the one-row eval sets"""

    def __init__(self, scope, out_dir):
        self.sc = SA.Scorer(scope, out_dir)
        self.torch = self.sc.torch

    def target(self, key, k):
        T = self.sc.target(key)
        rec, _ = SA.record_of(key)
        hp, n_ho = held_prompts(rec, k)
        tk, an = T["sets"]["ho"]
        assert n_ho == int(tk.shape[0]), (key, n_ho, int(tk.shape[0]))
        for p in hp:
            assert int(an[p["h"]]) == p["anchor"], (key, p["h"])
        s = rec["strong"]
        T["pp_prompts"] = {"train": [dict(p, q=q, row=q, peak=float(s["peak"][p["j"]]))
                                     for q, p in enumerate(T["prompts"][:k])],
                           "held": [dict(p, q=q, row=p["h"]) for q, p in enumerate(hp)]}
        return T

    @staticmethod
    def one_row(T, pset, row):
        """T with a single eval set "pp" = row `row` of the set behind pset (home for train, ho for held)"""
        nm = "home" if pset == "train" else "ho"
        tk, an = T["sets"][nm]
        Tq = dict(T)
        Tq["sets"] = {"pp": (tk[row:row + 1], an[row:row + 1])}
        Tq["pp"] = {f: T[nm][f][row:row + 1] for f in ("a_pos", "e0", "eM", "eN")}
        return Tq

    def capture(self, T, pset, row, members):
        """clean codes at the member sites (member columns only), positions 0..anchor; + the target's anchor read"""
        torch = self.torch
        from model.hooks import multi_patch
        nm = "home" if pset == "train" else "ho"
        tk, an = T["sets"][nm]
        anchor = int(an[row])
        toks = tk[row:row + 1, : anchor + 1]
        idx = {s: sorted(v) for s, v in members.items() if v}
        bad = [s for s in idx if s not in T["ups_real"]]
        assert not bad, bad
        it = {s: torch.tensor(v, dtype=torch.long, device=self.sc.dev) for s, v in idx.items()}
        got, read = {}, {}

        def tr(l, k, x):
            if (l, k) == T["site"]:
                read["v"] = float(torch.relu(x[0, anchor] @ T["w"].to(x.dtype) + T["b"].to(x.dtype)))
                return None
            if (l, k) in it:
                lat, _ = self.sc.bank.encode(x, k, l)
                got[(l, k)] = (idx[(l, k)], lat[0][:, it[(l, k)]].float().cpu())
            return None
        inf = self.sc.inf
        inf.disable_compile()
        try:
            with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                inf.forward(toks, patcher=lambda model: multi_patch(model, tr), grad_enabled=False,
                            return_activations=False, tokenize_final=False)
        finally:
            inf.enable_compile()
        return got, anchor, read.get("v")

    def score(self, u, Tq):
        rows = self.sc.unit(u, Tq)
        assert len(rows) == 1 and rows[0]["n_rows"] == 1
        return rows[0]


def unit(base, method, alphas, members, latpos, k=None):
    u = dict(base, method=method, alphas=alphas, members=members, latpos=latpos, j=None, prompt_tokens=None,
             n_latent=n_latent(members), n_latpos=None if latpos is None else len(latpos), topk=k)
    u["uid"] = "pp|%s|%s|%s|%s|%s%d%s" % (method, base["label"], base["dir"], base["key"], base["pset"], base["q"],
                                         "" if k is None else "|k%d" % k)
    return u


def ge_job(u, tokens):
    l, k, i = SA.parse_key(u["key"])
    dk, dl = ("z", i + SA.D_SAE) if k == "z2" else (k, i)
    nodes = ([list(n) for n in ge_nodes(u["latpos"])] if u["latpos"] is not None else
             [list(n) for n in SA.ge_job_nodes(dict(latpos=None, members=u["members"]))])
    return dict(job=u["uid"], uid=u["uid"], key=u["key"], j=u["row"], method=u["method"], layer=l, dict_kind=dk,
                dict_latent=dl, tokens=[int(t) for t in tokens], nodes=nodes)


# ------------------------------------------------------------------------------------------------ summary (CPU)
def summarize(out_dir, baseline_rows=None):
    rows = SA.jl(out_dir / "rows.jsonl")
    grec = {}
    for p in sorted(out_dir.glob("ge_recovery*.jsonl")):
        for r in SA.jl(p):
            if r.get("status") == "ok":
                grec[r["uid"]] = r
    for r in rows:
        g = grec.get(r["uid"])
        r["ge_rec"] = None if g is None else {f: g["rec"][f] for f in ("interp", "nonerr", "total", "bias", "err")}
    checks = SA.jl(out_dir / "check.jsonl")
    base = []
    if baseline_rows:
        for r in SA.jl(baseline_rows):
            if r["eval_set"] == "home" and r["method"] in ("ge_hier", "ge_posthoc", "dunefsky"):
                base.append(r)
        cov = {}
        for r in base:                     # rows scored before coverage_ours existed: as score_arena_a.summarize does
            if r["method"] == "dunefsky" and r.get("coverage_ours") is None and Path(r["src"]).exists():
                if r["src"] not in cov:
                    import torch
                    cov[r["src"]] = SA.dunefsky_coverage_ours(torch.load(r["src"], weights_only=False))
                r["coverage_ours"] = cov[r["src"]]
    f = lambda v, d=2: "-" if v is None else ("%.*f" % (d, v))  # noqa: E731
    curves = []
    for r in rows:
        rec = r.get("ge_rec") or {}
        knob = r.get("topk") if r["method"].startswith("wcm") else r.get("knob_value")
        curves.append(dict(method=r["method"], source="perprompt", label=r.get("label"), key=r["key"], pset=r["pset"],
                           q=r["q"], row=r["row"], knob=knob, n_latent=r["n_latent"], n_latpos=r["n_latpos"],
                           vacuous=r["vacuous"], pass_a=r["pass_a"], pass_b=r["pass_b"],
                           rec_interp=rec.get("interp"), rec_nonerr=rec.get("nonerr"), rec_total=rec.get("total"),
                           coverage=None, coverage_ours=None, **{m: r[m] for m in METRICS}))
    for r in base:
        rec = r.get("ge_rec") or {}
        curves.append(dict(method=r["method"], source="score_arena_a", label=None, key=r["key"], pset="train", q=r["j"],
                           row=r["j"], knob=r["knob_value"], n_latent=r["n_latent"], n_latpos=r["n_latpos"],
                           vacuous=r["vacuous"], pass_a=r["pass_a"], pass_b=r["pass_b"],
                           rec_interp=rec.get("interp"), rec_nonerr=rec.get("nonerr"), rec_total=rec.get("total"),
                           coverage=r.get("coverage"), coverage_ours=r.get("coverage_ours"),
                           **{m: r[m] for m in METRICS}))
    md = ["# Arena A per-prompt test (score_perprompt.py, A8b)", "",
          "One prompt per row; Z / A / C / nec as score_arena_a's home scope; **nec b = all upstream errors frozen at "
          "clean (primary)**; pass = Z, A, C in [0.75, 1.5] and nec >= 0.9; rec = Ge leaf-sum recovery / a_t "
          "(interp / nonerr / total) of the unit's nodes at their firing positions. train = the baselines' prompts "
          "(WCM was fitted on them); held = held-out strongest rows WCM never saw.", ""]
    if checks:
        worst = max(c["max_abs"] for c in checks)
        md += ["## Exactness check (restriction to ALL members x positions vs score_arena_a's home scope on prompt q)",
               "", "| target | arm | prompt | n latent | max abs diff over a_pos, e0, eM, eN, raw reads, Z..nec_b_tc |",
               "|---|---|---|---|---|"]
        for c in checks:
            md.append("| %s | %s | %d | %d | %.1e |" % (c["key"], c["label"], c["q"], c["n_latent"], c["max_abs"]))
        md += ["", "**max = %.2e (%s)**" % (worst, "EXACT" if worst == 0.0 else "PASS <= 1e-6" if worst <= 1e-6
                                            else "FAIL"), ""]
    fc = [r for r in rows if r["method"] == "wcm_fire" and r.get("fire_count_ref") is not None]
    if fc:
        bad = [r for r in fc if (r["n_latent"], r["n_latpos"]) != tuple(r["fire_count_ref"])]
        md += ["Firing sizes vs fire_count.json on the same prompt rows: %d of %d identical." % (len(fc) - len(bad),
                                                                                                 len(fc)), ""]
    md += ["## Per prompt", "",
           "| method | target | arm | set | q (row) | k | n latent | n (lat,pos) | a_pos | Z | A | C | nec a | **nec b** | "
           "nec b tc | pass a | **pass b** | rec interp | rec nonerr | rec total |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"wcm_full": 0, "wcm_fire": 1, "wcm_fire_top": 2}
    for r in sorted(rows, key=lambda r: (r["key"], str(r.get("label")), r["pset"], r["q"], order.get(r["method"], 3),
                                         r["method"], -(r.get("topk") or 0), r.get("knob_index") or 0)):
        rec = r.get("ge_rec") or {}
        kn = r.get("topk") if r["method"].startswith("wcm") else r.get("knob_value")
        md.append("| %s | %s | %s | %s | %d (%d) | %s | %d | %s | %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s "
                  "| %s |" % (r["method"], r["key"], r.get("label") or "-", r["pset"], r["q"], r["row"],
                              "-" if kn is None else ("%.3g" % kn), r["n_latent"],
                              "-" if r["n_latpos"] is None else r["n_latpos"], r["a_pos"], f(r["Z"]), f(r["A"]),
                              f(r["C"]), f(r["nec_a"]), f(r["nec_b"]), f(r["nec_b_tc"]), r["pass_a"], r["pass_b"],
                              f(rec.get("interp")), f(rec.get("nonerr")), f(rec.get("total"))))
    md.append("")
    summ = {}
    groups = {}
    for c in curves:
        if c["method"] == "wcm_fire_top":
            continue
        groups.setdefault((c["method"], c["pset"], c["source"]), []).append(c)
    md += ["## Medians per method and prompt set (non-vacuous prompts; baselines = score_arena_a home rows, every knob "
           "point; DIFFERENT TARGETS from the WCM rows on the smoke)", "",
           "| method | set | source | points | vacuous | n latent | Z | A | C | nec a | nec b | nec b tc | pass a | "
           "pass b | rec interp | rec total | coverage | coverage ours |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, ps, src), cs in sorted(groups.items()):
        nv = [c for c in cs if not c["vacuous"]]

        def med(fl):
            v = [c[fl] for c in nv if c[fl] is not None]
            return stats.median(v) if v else None
        s = dict(points=len(cs), vacuous=len(cs) - len(nv), n_latent=med("n_latent"),
                 **{x: med(x) for x in METRICS + ("rec_interp", "rec_total", "coverage", "coverage_ours")},
                 pass_a=sum(bool(c["pass_a"]) for c in nv), pass_b=sum(bool(c["pass_b"]) for c in nv),
                 keys=sorted({c["key"] for c in cs}))
        summ["%s|%s|%s" % (m, ps, src)] = s
        md.append("| %s | %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %d | %d | %s | %s | %s | %s |" % (
            m, ps, src, s["points"], s["vacuous"], f(s["n_latent"], 0), f(s["Z"]), f(s["A"]), f(s["C"]),
            f(s["nec_a"]), f(s["nec_b"]), f(s["nec_b_tc"]), s["pass_a"], s["pass_b"], f(s["rec_interp"]),
            f(s["rec_total"]), f(s["coverage"]), f(s["coverage_ours"])))
    md.append("")
    cols = ["method", "source", "label", "key", "pset", "q", "row", "knob", "n_latent", "n_latpos", "vacuous",
            "pass_a", "pass_b"] + list(METRICS) + ["rec_interp", "rec_nonerr", "rec_total", "coverage", "coverage_ours"]
    csv = [",".join(cols)] + [",".join("" if c[x] is None else str(c[x]) for x in cols) for c in curves]
    for name, text in (("summary.md", "\n".join(md) + "\n"), ("summary.json", json.dumps(summ, indent=1) + "\n"),
                       ("curves.jsonl", "".join(json.dumps(c) + "\n" for c in curves)),
                       ("curves.csv", "\n".join(csv) + "\n")):
        tmp = out_dir / (name + ".tmp")
        tmp.write_text(text)
        tmp.replace(out_dir / name)
    print("\n".join(md))


# ------------------------------------------------------------------------------------------------ main
def fire_count_ref(path):
    """{(key, record row j): (mem_any, mem_pos)} from fire_count.json (train prompts) for the size cross-check"""
    if not Path(path).exists():
        return {}
    d = json.load(open(path))
    out = {}
    for res in d.get("results", []):
        for which in ("train", "held"):
            for r in res["prompts"].get(which, []):
                out[(res["key"], which, int(r["row"]))] = (int(r["mem_any"]), int(r["mem_pos"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--scope", default=os.environ.get("ARENA_A_SCOPE", "a1"), choices=("a1", "a2"))
    ap.add_argument("--wcm", default="", help="label=dir,... (label l<lambda>), as score_arena_a")
    ap.add_argument("--topk-arms", default="", help="labels that get the top-k curve [default: the first --wcm label]")
    ap.add_argument("--keys", default="")
    ap.add_argument("--sets", default="train,held")
    ap.add_argument("--n-prompts", type=int, default=K_PROMPTS, help="prompts per set (<= K)")
    ap.add_argument("--check", action="store_true", help="exactness check per train prompt (keep-all restriction)")
    ap.add_argument("--ge-held", default="", help="run_ge.py output on the held prompts (--emit-heldout-prompts)")
    ap.add_argument("--dunefsky-held", default="", help="run_dunefsky.py output on the held prompts")
    ap.add_argument("--tau-stride", type=int, default=1)
    ap.add_argument("--fire-count", default=str(HERE / "out" / "panel" / "fire_count" / "fire_count.json"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-epoch", type=float, default=0, help="no new target after this unix time")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--baseline-rows", default="", help="score_arena_a rows.jsonl (home rows = the train baselines)")
    ap.add_argument("--emit-heldout-prompts", default="", help="write DIR/prompts.jsonl (held prompts) and exit")
    a = ap.parse_args()
    out_dir = Path(a.out)
    if a.summary:
        out_dir.mkdir(parents=True, exist_ok=True)
        return summarize(out_dir, a.baseline_rows or None)
    os.environ["ARENA_A_SCOPE"] = a.scope
    sys.path.insert(0, str(SA.ROOT / "src"))
    import torch
    torch.set_num_threads(a.threads)
    keys = [k for k in a.keys.split(",") if k] or None
    sets = [s for s in a.sets.split(",") if s]
    assert all(s in SETS for s in sets), sets
    if a.emit_heldout_prompts:
        import wcm_tcz as WT
        WT.register()
        assert keys, "--emit-heldout-prompts needs --keys"
        return emit_heldout_prompts(a.emit_heldout_prompts, keys, a.n_prompts)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        import wcm_tcz as WT
        WT.register()
        pp = None
    else:
        pp = PerPrompt(a.scope, out_dir)
    specs = [tuple(x.split("=", 1)) for x in a.wcm.split(",") if x]
    topk_arms = set(x for x in a.topk_arms.split(",") if x) or ({specs[0][0]} if specs else set())
    wus = SA.wcm_units(specs, set(keys) if keys else None, a.scope)
    bus = []
    if a.ge_held:
        bus += SA.ge_units(a.ge_held, set(keys) if keys else None, a.scope, a.tau_stride)
    if a.dunefsky_held:
        bus += SA.dunefsky_units(a.dunefsky_held, set(keys) if keys else None, a.scope)
    done = {r["uid"] for r in SA.jl(out_dir / "rows.jsonl")}
    cdone = {(r["key"], r["label"], r["q"]) for r in SA.jl(out_dir / "check.jsonl")}
    jdone = {r["job"] for r in SA.jl(out_dir / "ge_jobs.pp.jsonl")}
    by_t = {}
    for u in wus + bus:
        by_t.setdefault(u["key"], []).append(u)
    n_est = sum(len(sets) * a.n_prompts * (2 + (8 if u["label"] in topk_arms else 0)) for u in wus)
    print("score_perprompt: scope %s, %d WCM circuits (%s; top-k arms %s), %d held-out baseline units, %d targets, "
          "sets %s x %d prompts; ~%d WCM units (top-k grid sized at run time); %d rows done" % (
              a.scope, len(wus), sorted({u["label"] for u in wus}), sorted(topk_arms), len(bus), len(by_t), sets,
              a.n_prompts, n_est, len(done)), flush=True)
    for key, us in sorted(by_t.items()):
        print("  %-12s %s" % (key, ", ".join("%s %s n %d" % (u["method"], u.get("label") or u["knob_value"],
                                                              u["n_latent"]) for u in us[:6])), flush=True)
    if a.dry_run:
        return
    fcr = fire_count_ref(a.fire_count)
    fh = open(out_dir / "rows.jsonl", "a")
    fj = open(out_dir / "ge_jobs.pp.jsonl", "a")
    fc = open(out_dir / "check.jsonl", "a")
    fmt = lambda v: "-" if v is None else "%.3f" % v  # noqa: E731
    for key, us in sorted(by_t.items()):
        if a.stop_epoch and time.time() >= a.stop_epoch:
            print("stop epoch reached: not starting %s" % key, flush=True)
            break
        t0 = time.time()
        T = pp.target(key, a.n_prompts)
        print("[%s] %s target ready (%.0fs)" % (time.strftime("%H:%M:%S"), key, time.time() - t0), flush=True)
        for u0 in us:
            if u0["method"] != "wcm":
                continue
            for pset in sets:
                for pr in T["pp_prompts"][pset]:
                    q, row = pr["q"], pr["row"]
                    base = dict(key=key, label=u0["label"], dir=Path(u0["src"]).parents[2].name, src=u0["src"],
                                knob="lambda", knob_value=u0["knob_value"], pset=pset, q=q, row=row,
                                rec_row=pr["j"], anchor=pr["anchor"], peak=pr["peak"])
                    want = [unit(base, "wcm_full", u0["alphas"], u0["members"], None)]
                    planned = {want[0]["uid"]}
                    need_check = a.check and pset == "train" and (key, u0["label"], q) not in cdone
                    t1 = time.time()
                    codes, anchor, read = pp.capture(T, pset, row, u0["members"])
                    lp = firing_latpos(codes, anchor)
                    al_f, mem_f = restrict(u0["alphas"], lp)
                    uf = unit(base, "wcm_fire", al_f, mem_f, lp)
                    uf["fire_count_ref"] = fcr.get((key, pset, pr["j"]))
                    uf["read_clean"] = read
                    want.append(uf)
                    if u0["label"] in topk_arms:
                        amax = act_max(codes, anchor)
                        order = topk_order(u0["alphas"], amax, sorted({(l, k, i) for l, k, _p, i in lp}))
                        for kk in k_grid(len(order)):
                            lpk = latpos_of(lp, order[:kk])
                            al_k, mem_k = restrict(u0["alphas"], lpk)
                            want.append(unit(base, "wcm_fire_top", al_k, mem_k, lpk, kk))
                    Tq = pp.one_row(T, pset, row)
                    cache = {}
                    for u in want:
                        planned.add(u["uid"])
                        if u["uid"] in done:
                            continue
                        sg = sig(u["members"], u["alphas"])
                        if sg in cache:
                            r = dict(cache[sg], **SA.jsonable(u), reused=True)
                        else:
                            r = pp.score(u, Tq)
                            r.pop("eval_set", None)
                            cache[sg] = r
                        r["eval_set"] = "pp"
                        fh.write(json.dumps(r) + "\n")
                        done.add(u["uid"])
                        if u["uid"] not in jdone:
                            fj.write(json.dumps(ge_job(u, pr["tokens"])) + "\n")
                            jdone.add(u["uid"])
                    fh.flush()
                    fj.flush()
                    if need_check:
                        # keep-all restriction through restrict(), scored on the one-row set, vs score_arena_a's home
                        # scope on prompt q (Scorer.unit with j = q on the full 8-row home set)
                        al_c, mem_c = restrict(u0["alphas"], all_latpos(u0["alphas"], anchor + 1))
                        r_c = pp.score(unit(base, "wcm_check", al_c, mem_c, None), Tq)
                        Th = dict(T, sets={"home": T["sets"]["home"]})
                        r_h = [x for x in pp.sc.unit(dict(u0, j=q, prompt_tokens=None), Th) if x["eval_set"] == "home"]
                        assert len(r_h) == 1 and r_h[0]["n_rows"] == 1
                        r_h = r_h[0]
                        flds = ("a_pos", "e0", "eM_topk", "eN_topk", "free0_raw", "freeM_raw", "freeN_raw", "rm_a_raw",
                                "rm_b_raw", "rm_btc_raw", "clean_diag") + METRICS
                        diffs = {x: (None if r_c[x] is None or r_h[x] is None else abs(r_c[x] - r_h[x])) for x in flds}
                        nn = [x for x in flds if (r_c[x] is None) != (r_h[x] is None)]
                        mx = max([v for v in diffs.values() if v is not None] or [0.0])
                        fc.write(json.dumps(dict(key=key, label=u0["label"], q=q, n_latent=u0["n_latent"],
                                                 max_abs=mx if not nn else float("inf"), none_mismatch=nn,
                                                 diffs=diffs)) + "\n")
                        fc.flush()
                        cdone.add((key, u0["label"], q))
                        print("    check %s q%d: max |diff| %.1e%s" % (key, q, mx, (" None-mismatch %s" % nn) if nn
                                                                        else ""), flush=True)
                    print("  %s %s %s q%d row %d: fire %d / %d latents, %d (lat,pos); clean read %.3f vs peak %.3f; "
                          "%d units %.1fs" % (key, u0["label"], pset, q, row, uf["n_latent"], u0["n_latent"],
                                              uf["n_latpos"], read, pr["peak"], len(want), time.time() - t1),
                          flush=True)
        # held-out baselines (their circuits built on the held prompts): prompt j of their file = held prompt q
        for u in us:
            if u["method"] == "wcm" or u["uid"] in done or "held" not in sets:
                continue
            hp = T["pp_prompts"]["held"]
            if u["j"] >= len(hp):
                continue
            pr = hp[u["j"]]
            assert u["prompt_tokens"] == pr["tokens"], (u["uid"], "held prompt mismatch")
            ub = dict(u, uid="pp|" + u["uid"], j=None, prompt_tokens=None, pset="held", q=u["j"], row=pr["row"],
                      label=None, topk=None)
            r = pp.score(ub, pp.one_row(T, "held", pr["row"]))
            r["eval_set"] = "pp"
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done.add(ub["uid"])
            print("  %-48s n %5d | Z %s A %s C %s nec a %s b %s" % (ub["uid"], ub["n_latent"], fmt(r["Z"]), fmt(r["A"]),
                                                                     fmt(r["C"]), fmt(r["nec_a"]), fmt(r["nec_b"])),
                  flush=True)
    print("score_perprompt: done", flush=True)


if __name__ == "__main__":
    main()
