"""Experiment 081 report (CPU): every number the paper paragraph would quote, and the registered verdicts T1-T5
(README, re-registered 2026-10-21).

Sections:
  0  competence (probe/<MODEL>/competence.json): every task, held-out metric, share correct, counterfactual gap, top-1
  1  per competent task: the size-faithfulness curve of W and U (lambda, size, task Z / A / C, necessity, pass,
     accuracy recovered by the circuit alone, Z at alpha = 1), the random null at each W lambda, and for
     generated-contrast tasks the W_ZA sensitivity arm
  2  claim 2: size to reach task Z >= 0.8 per arm and U / W; smallest passing size per arm; U at W's sizes
  3  claim 1 null: random circuits with fitted coefficients
  3b sensitivity (generated-contrast tasks): W (Z + C + A) vs W_ZA (Z + A), each under the full rule and the Z + A rule
  4  structure (T4): IOI heads (Wang et al.) and greater-than heads / MLPs (Hanna et al.) on the primary W circuit,
     via the attention members' exact per-head drive; agreement checks (secondary)
  5  link to latent circuits (T5, descriptive; TuringLLM only, since the 062 production targets are TuringLLM latents)
  6  verdicts and the paper numbers

  MODEL=gpt2-small PYTHONPATH=src python -X utf8 experiments/081-task-circuits/report.py
  env: MODEL  OUT (out/<MODEL>)  OUT_PROBE (probe/<MODEL>)  TOP (20)  SRC_062  NO_062=1  JSON (OUT/report.json)
"""
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task_common as C  # noqa: E402

OUT = Path(os.environ.get("OUT", str(HERE / "out" / C.MODEL)))
OUT_PROBE = Path(os.environ.get("OUT_PROBE", str(HERE / "probe" / C.MODEL)))
TOP = int(os.environ.get("TOP", 20))
SRC_062 = Path(os.environ.get("SRC_062", str(C.EXP / "062-h100-protocol-v1" / "out_full" / "out")))


# ------------------------------------------------------------------------------------------- loading
def read_jsonl(path):
    if not Path(path).exists():
        return []
    op = gzip.open if str(path).endswith(".gz") else open
    out = []
    with op(path, "rt") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


def load():
    """(eval rows, the last per (task, arm, lambda, draw); profiles {(task, lambda): members}; competence)."""
    rows = {}
    for r in read_jsonl(OUT / "eval.jsonl"):
        rows[(r["task"], r["arm"], r["lam"], int(r.get("draw", -1)))] = r
    prof = {}
    for r in read_jsonl(OUT / "profiles.jsonl.gz"):
        prof[(r["task"], r["lam"])] = r["members"]
    comp = {}
    if (OUT_PROBE / "competence.json").exists():
        comp = json.loads((OUT_PROBE / "competence.json").read_text())["tasks"]
    return list(rows.values()), prof, comp


def task_curves(rows):
    """{task: {"W": curve, "U": curve, "W_ZA": curve, "RND": {lam_tag: [rows]}}}."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["task"]][r["arm"]].append(r)
    out = {}
    for task, arms in by.items():
        rnd = defaultdict(list)
        for r in arms.get("RND", []):
            if not (r.get("error") or r.get("skip")):
                rnd[r["lam"]].append(r)
        out[task] = dict(W=C.curve(arms.get("W", [])), U=C.curve(arms.get("U", [])),
                         W_ZA=C.curve(arms.get("W_ZA", [])), RND=dict(rnd))
    return out


def f2(v):
    return "%.2f" % v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf") \
        else ("inf" if v == float("inf") else "-")


def f0(v):
    return "%.0f" % v if isinstance(v, (int, float)) and v == v and v != float("inf") else ("inf" if v == float("inf")
                                                                                         else "-")


# ------------------------------------------------------------------------------------------- 062 links (TuringLLM)
def production_targets(src=SRC_062):
    d = Path(src) / "main" / "circuits"
    return {p.stem for p in d.glob("*.pt")} if d.exists() else set()


def latent_circuit_members(key, src=SRC_062):
    import torch
    p = Path(src) / "main" / "circuits" / ("%s.pt" % key)
    if not p.exists():
        return None
    c = torch.load(p, weights_only=False, map_location="cpu")
    if c is None:
        return None
    return {(int(n.metadata["feature_id"].layer), str(n.metadata["feature_id"].kind),
             int(n.metadata["feature_id"].index)) for n in c.nodes.values() if n.metadata.get("role") != "seed"}


def active_set(task):
    p = OUT / "active" / ("%s.json.gz" % task)
    if not p.exists():
        return None
    d = json.loads(gzip.open(p, "rt").read())
    return {(int(k.split("/")[0]), k.split("/")[1], int(i)) for k, v in d.items() for i in v}


# ------------------------------------------------------------------------------------------- main
def main():
    rows, prof, comp = load()
    R = dict(model=C.MODEL)
    print("MODEL %s\n" % C.MODEL)
    print("== 0. competence (held-out items; rule: mean > 0, >= 70% correct, and clean - counterfactual > 0 where the "
          "dataset has counterfactuals)")
    print("%-44s %-12s %5s %9s %8s %8s %6s %6s  %s" % ("task", "family", "n", "mean", "correct", "cf gap", "top-1",
                                                      "expect", "verdict"))
    for name, r in comp.items():
        h = r["held"]
        print("%-44s %-12s %5d %9.3f %7.0f%% %8s %6s %6s  %s" % (
            name, r["family"], h["n"], h["mean_clean"], 100 * h["frac_correct"], f2(h.get("cf_gap")),
            f2(h.get("top1")), {True: "yes", False: "no", None: "-"}[r.get("expect")],
            "COMPETENT" if r["competent"] else "not"))
    R["competence"] = {k: dict(competent=v["competent"], mean=v["held"]["mean_clean"],
                               frac_correct=v["held"]["frac_correct"], cf_gap=v["held"].get("cf_gap"),
                               top1=v["held"].get("top1"), n=v["held"]["n"], contrast=v.get("contrast"))
                       for k, v in comp.items()}
    if not rows:
        print("\n(no fits in %s yet)" % OUT)
        _dump(R)
        return
    TC = task_curves(rows)
    print("\n== 1. size-faithfulness per task (held-out; task metric; * = passes the v1 rule; za = passes Z + A)")
    for task, cv in sorted(TC.items()):
        print("\n-- %s (contrast: %s)" % (task, C.task_spec(task)["contrast"]))
        print("   %-5s %-7s %7s %6s %6s %6s %6s %6s %6s %6s" % ("arm", "lambda", "n", "Z", "A", "C", "nec", "acc",
                                                             "Z(a=1)", "pass"))
        for arm in ("W", "U", "W_ZA"):
            for p in cv[arm]:
                print("   %-5s %-7s %7d %6s %6s %6s %6s %6s %6s %6s" % (
                    arm, C.lam_tag(p["lam"]), p["n"], f2(p["Z"]), f2(p["A"]), f2(p["C"]), f2(p["nec"]), f2(p["acc"]),
                    f2(p["Z_a1"]), ("*" if p["passed"] else "") + (" za" if p["passed_ZA"] else "")))
        for lt, rs in sorted(cv["RND"].items(), key=lambda x: float(x[0])):
            zs = sorted(C.num(r.get("task_free0")) for r in rs if C.num(r.get("task_free0")) is not None)
            print("   RND   %-7s %7s draws %d: Z median %s [min %s, max %s], pass %d/%d" % (
                lt, f0(rs[0].get("n")), len(rs), f2(zs[len(zs) // 2] if zs else None), f2(zs[0] if zs else None),
                f2(zs[-1] if zs else None), sum(bool(r.get("task_pass")) for r in rs), len(rs)))

    print("\n== 2. claim 2: coefficients keep task circuits small (U / W)")
    print("%-44s %8s %10s %8s %8s %8s %8s %10s" % ("task", "W@0.8", "U@0.8", "ratio", "W pass", "U pass", "ratio",
                                                   "U@Wpass"))
    ratios = {}
    for task, cv in sorted(TC.items()):
        sr = C.size_ratio(cv["W"], cv["U"])
        uw = C.u_at_w_sizes(cv["W"], cv["U"]) if cv["U"] else []
        at = [u for u in uw if u["W_pass"] and u["n"] == sr["pass_W"]]
        sr["u_fails_at_w_pass"] = (bool(at[0]["U_pass"] is None or at[0]["U_pass"] < 0.5) if at else None)
        ratios[task] = sr
        print("%-44s %8s %10s %8s %8s %8s %8s %10s" % (task, f0(sr["n_reach_W"]), f0(sr["n_reach_U"]) + (
            "" if sr["status_U"] != "never" else ">%s" % f0(sr["u_max_n"])), f2(sr["reach_ratio"]), f0(sr["pass_W"]),
            f0(sr["pass_U"]), f2(sr["pass_ratio"]), {True: "fails", False: "passes", None: "-"}[sr["u_fails_at_w_pass"]]))
    R["claim2"] = ratios

    print("\n== 3. random null (claim 1): random supports matched to W, coefficients fitted")
    rnd_pass, below = [], {}
    for task, cv in sorted(TC.items()):
        wz = {C.lam_tag(p["lam"]): p["Z"] for p in cv["W"]}
        diffs = []
        for lt, rs in cv["RND"].items():
            rnd_pass += [bool(r.get("task_pass")) for r in rs]
            zs = sorted(C.num(r.get("task_free0")) for r in rs if C.num(r.get("task_free0")) is not None)
            if zs and wz.get(lt) is not None:
                diffs.append(zs[len(zs) // 2] < wz[lt])
        if diffs:
            below[task] = sum(diffs) * 2 > len(diffs)
            n_p = sum(bool(r.get("task_pass")) for rs in cv["RND"].values() for r in rs)
            n_r = sum(len(rs) for rs in cv["RND"].values())
            print("  %-44s draws %3d  pass %d  RND median Z below W's in %d / %d lambdas" % (task, n_r, n_p, sum(diffs),
                                                                                       len(diffs)))
    v3, s3 = C.verdict_t3(rnd_pass, below)
    print("  overall: %d random circuits, pass rate %s (95%% upper %s)" % (s3.get("n", 0), f2(s3.get("pass_rate")),
                                                                         f2(s3.get("upper95"))))

    print("\n== 3b. sensitivity: generated-contrast tasks, W (trained Z + C + A) vs W_ZA (trained Z + A)")
    sens = {}
    for task, cv in sorted(TC.items()):
        if C.task_spec(task)["contrast"] != "generated":
            continue
        s = {arm: dict(full=any(p["passed"] for p in cv[arm]), za=any(p["passed_ZA"] for p in cv[arm]),
                       smallest_full=C.smallest_passing(cv[arm]),
                       smallest_za=min([p["n"] for p in cv[arm] if p["passed_ZA"]], default=None))
             for arm in ("W", "W_ZA")}
        sens[task] = s
        print("  %-44s W: passes full %s / Z+A %s (smallest %s / %s) | W_ZA: passes full %s / Z+A %s (smallest %s / "
              "%s)" % (task, s["W"]["full"], s["W"]["za"], s["W"]["smallest_full"], s["W"]["smallest_za"],
                       s["W_ZA"]["full"], s["W_ZA"]["za"], s["W_ZA"]["smallest_full"], s["W_ZA"]["smallest_za"]))
    R["sensitivity"] = sens

    print("\n== 4. structure vs published circuits (primary W circuit: smallest passing; else best worst-ablation)")
    checks, primary = {}, {}
    for task, cv in sorted(TC.items()):
        lam, ok = C.primary_lambda(cv["W"])
        if lam is None:
            continue
        lt = C.lam_tag(lam)
        primary[task] = dict(lam=lam, passing=ok, n=next(p["n"] for p in cv["W"] if p["lam"] == lam))
        pr = prof.get((task, lt))
        if pr is None:
            continue
        fam = C.task_spec(task)["family"]
        rc = C.role_contrib(pr)
        tot = sum(rc.values()) or 1.0
        by_layer = defaultdict(float)
        for m in pr:
            by_layer[(m["layer"], m["kind"])] += abs(m.get("contrib_anchor") or 0.0)
        ta = sum(by_layer.values()) or 1.0
        print("  %-44s lambda %s (%s, n %d) | |attribution| by role: %s" % (
            task, lt, "passes" if ok else "best, does not pass", primary[task]["n"],
            ", ".join("%s %.0f%%" % (k, 100 * v / tot) for k, v in sorted(rc.items(), key=lambda x: -x[1])[:6])))
        top_sites = sorted(by_layer.items(), key=lambda x: -x[1])[:8]
        print("      anchor attribution, top sites: %s" % " ".join("L%d.%s %.0f%%" % (s[0], s[1], 100 * v / ta)
                                                                    for s, v in top_sites))
        if fam in ("ioi", "greater_than"):
            hw = C.head_weights(pr, C.ANCHOR)
            top_h = sorted(hw.items(), key=lambda x: -x[1])[:10]
            tw = sum(hw.values()) or 1.0
            print("      attn-member attribution at the anchor by head: %s" % " ".join(
                "%d.%d %.0f%%" % (h[0], h[1], 100 * v / tw) for h, v in top_h))
        if fam in ("ioi", "greater_than", "sva"):
            checks[task] = C.structure_checks(fam, task, pr)
            for name, c in checks[task].items():
                ci = (" [%s, %s]" % (f2(c.get("lo")), f2(c.get("hi")))) if c.get("lo") is not None else ""
                extra = (" (share %s vs base %s)" % (f2(c.get("share")), f2(c.get("base")))) if "base" in c else ""
                print("      %-32s %s%s%s -> %s%s" % (name, f2(c.get("value")), ci, extra,
                                                      {True: "holds", False: "FAILS", None: "not evaluable"}[c["held"]],
                                                      "" if c.get("primary", True) else " (secondary)"))
    v4, s4 = C.verdict_t4(checks)
    v4b, s4b = C.verdict_t4(checks, primary_only=False)
    R["structure"] = checks
    R["primary"] = primary

    print("\n== 5. link to the 062 latent circuits (T5, descriptive)")
    links = {}
    if C.MODEL != "turingllm":
        print("  not applicable on %s: the 062 production targets and their latent circuits are TuringLLM latents. "
              "Descriptive instead: each primary W circuit's top members by |anchor attribution| (Neuronpedia ids "
              "gpt2-small/<layer>-<res_post|res_mid|att|mlp>_32k-oai/<latent>)" % C.MODEL)
        np_kind = {"resid": "res_post", "resid_mid": "res_mid", "attn": "att", "mlp": "mlp"}
        for task, pinfo in sorted(primary.items()):
            pr = prof.get((task, C.lam_tag(pinfo["lam"])))
            if not pr:
                continue
            top = sorted(pr, key=lambda m: -abs(m.get("contrib_anchor") or 0.0))[:min(TOP, 10)]
            links[task] = [dict(layer=m["layer"], kind=m["kind"], latent=m["latent"], alpha=m["alpha"],
                                contrib_anchor=m.get("contrib_anchor"),
                                neuronpedia="gpt2-small/%d-%s_32k-oai/%d" % (m["layer"], np_kind.get(m["kind"], m["kind"]),
                                                                            m["latent"])) for m in top]
            print("  %-44s %s" % (task, " ".join("%d.%s.%d(%s)" % (m["layer"], m["kind"], m["latent"],
                                                                  f2(m.get("contrib_anchor"))) for m in top)))
    else:
        targets = production_targets() if os.environ.get("NO_062") != "1" else set()
        if not targets:
            print("  (no 062 production circuits under %s; skipped)" % SRC_062)
        for task, pinfo in sorted(primary.items()):
            pr = prof.get((task, C.lam_tag(pinfo["lam"])))
            if not pr or not targets:
                continue
            top = sorted(pr, key=lambda m: -abs(m.get("contrib_anchor") or 0.0))[:TOP]
            top_k = [(m["layer"], m["kind"], m["latent"]) for m in top]
            members = {(m["layer"], m["kind"], m["latent"]) for m in pr}
            lc = {}
            for m in top_k:
                k = C.member_key(m)
                if k in targets:
                    s = latent_circuit_members(k)
                    if s is not None:
                        lc[k] = s
            links[task] = L = C.link_to_latent_circuits(top_k, members, targets, lc, active_set(task))
            print("  %-44s %d/%d top members are production targets (base rates %s of members, %s of active "
                  "latents) | union %d latents: covers %s of the task circuit, %s of it in the task circuit (J %s)" % (
                      task, L["n_top_targets"], L["n_top"], f2(L["base_rate_members"]), f2(L["base_rate_active"]),
                      L["n_union"], f2(L["task_in_union"]), f2(L["union_in_task"]), f2(L["jaccard"])))
    R["t5"] = links

    print("\n== 6. registered predictions (2026-10-21)")
    passed_any = {t: any(p["passed"] for p in cv["W"]) for t, cv in TC.items()}
    v1 = C.verdict_t1(passed_any)
    v2, s2 = C.verdict_t2(ratios)
    print("  T1 competent tasks get a passing W circuit at some lambda: %s (%d / %d tasks)" % (
        v1, sum(passed_any.values()), len(passed_any)))
    print("  T2 U needs >= 2x the latents to reach Z 0.8 and fails at W's passing size: %s (%s)" % (v2, s2))
    print("  T3 random circuits with fitted coefficients fail: %s (%s)" % (v3, s3))
    print("  T4 structure matches the published GPT-2 circuits (IOI, greater-than): %s (%s); with the agreement "
          "checks: %s (%s)" % (v4, s4, v4b, s4b))
    print("  T5 (descriptive): %s" % ({t: "%d/%d" % (L["n_top_targets"], L["n_top"]) for t, L in links.items()}
                                      if C.MODEL == "turingllm" and links else
                                      "top members listed in section 5 (no 062 link on %s)" % C.MODEL))
    R["verdicts"] = dict(T1=v1, T2=[v2, s2], T3=[v3, s3], T4=[v4, s4], T4_all=[v4b, s4b])

    print("\n== paper numbers")
    comp_ok = [k for k, v in comp.items() if v["competent"]]
    print("  competent: %d of %d tasks (%s); not competent: %s" % (len(comp_ok), len(comp), ", ".join(comp_ok),
                                                                  ", ".join(k for k in comp if k not in comp_ok)))
    for t, cv in sorted(TC.items()):
        sr = ratios.get(t, {})
        pw = [p for p in cv["W"] if p["passed"]]
        best = min(pw, key=lambda p: p["n"]) if pw else None
        print("  %-44s smallest passing W: %s latents (Z %s A %s C %s nec %s, accuracy %s; at alpha = 1 Z %s) | "
              "U to reach Z 0.8: %s (x%s) | RND pass %s" % (
                  t, best["n"] if best else "none", f2(best and best["Z"]), f2(best and best["A"]),
                  f2(best and best["C"]), f2(best and best["nec"]), f2(best and best["acc"]), f2(best and best["Z_a1"]),
                  f0(sr.get("n_reach_U")), f2(sr.get("reach_ratio")),
                  "%d/%d" % (sum(bool(r.get("task_pass")) for rs in cv["RND"].values() for r in rs),
                             sum(len(rs) for rs in cv["RND"].values()))))
    _dump(R)


def _dump(R):
    path = Path(os.environ.get("JSON", str(OUT / "report.json")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(R, indent=1, default=lambda o: None if o != o else str(o)) + "\n")
    print("\nnumbers -> %s" % path)


if __name__ == "__main__":
    main()
