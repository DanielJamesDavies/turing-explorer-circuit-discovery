"""Table 2: exclusions accounting per model, from the target frame down to the scored circuits, with reasons.

Reads   public: 082 targets_<model>_manifest.json (eligible-latent frame per component), targets_<model>_main.txt,
                out_<model>/main/status.shard*.jsonl, contexts.shard*.jsonl (held-out slice sizes of failed targets),
                main/eval.shard*.jsonl
        TuringLLM: out_turingllm/lists/targets_tl8k.txt + quarantine.json (T2), out_turingllm/records/build.shard*
                (skip reasons of the record build), out_v2/main/status + eval (T3)
Writes  paper/tables/run2026_exclusions.tex, out/run2026_exclusions.csv (one row per model),
        out/run2026_excluded_targets.csv (one row per excluded target with its reason)

Steps (each target falls out at the first step it fails):
  targets      the frozen main list
  no record    TuringLLM: the record build skipped it (skip reason from build rows, n_top / n_contrast recorded);
               public: no status row at all
  quarantined  TuringLLM T2 quarantine list (public: none in the chain)
  skipped      status row with a skip (no circuit fitted)
  fit error    last status row carries an error (transient errors that were retried to ok are counted separately)
  no circuit   status ok but no held-out-strongest eval row
  vacuous      merge.vacuous on the held-out strongest row
  scored       the headline denominator
Mid-band: no mid row (thin: no mid-band pool, arm A) and mid vacuous, reported beside.

  python -X utf8 experiments/109-pod-run-analysis/t2_exclusions.py
"""
import json
from collections import Counter

import pandas as pd

import lib109 as L


def last_status(model):
    rows = L.read_rows(L.find(model, "out", "main"), "status")
    last, n_err_rows, err_seeds = {}, 0, set()
    for r in rows:
        if "error" in r:
            n_err_rows += 1
            err_seeds.add(r["seed"])
        prev = last.get(r["seed"])
        # an ok row is final; a later error row never replaces an ok one (retries run only on failed targets)
        if prev is None or not prev.get("ok") or r.get("ok"):
            last[r["seed"]] = r
    return last, n_err_rows, err_seeds


def contexts_info(model, seeds):
    out = {}
    for r in L.read_rows(L.find(model, "out", "main"), "contexts"):
        if r.get("seed") in seeds:
            out[r["seed"]] = dict(n_train=len(r.get("train_ids") or []), n_held_strong=len(r.get("held_strong_ids") or []),
                                  n_held_mid=len(r.get("held_mid_ids") or []))
    return out


def account(model):
    tg = L.targets_list(model)
    if tg is None:
        L.not_available("%s target list" % model)
        return None, []
    st, n_err_rows, err_seeds = last_status(model)
    if not st:
        L.not_available("%s main status rows" % model)
        return None, []
    ev = L.main_eval(model)
    strong = ev[ev.held == "strong"].set_index("seed") if len(ev) else pd.DataFrame()
    mid = ev[ev.held == "mid"].set_index("seed") if len(ev) else pd.DataFrame()
    excluded = []
    row = dict(model=L.LABEL[model], targets=len(tg))
    # eligible-latent frame (public manifests)
    man = L.find_file(model, "run", "targets_%s_manifest.json" % model) if model != "turingllm" else None
    if man:
        m = json.load(open(man, encoding="utf-8"))
        row["frame_latents"] = sum(c.get("frame", 0) for c in m.get("components", []))
        row["frame_components"] = len(m.get("components", []))
    # no record / quarantine (TuringLLM)
    build_skip, quarantined = {}, set()
    if model == "turingllm":
        for r in L.read_rows(L.find(model, "work", "records"), "build"):
            if r.get("skip"):
                build_skip[r["seed"]] = r
        q = L.find_file(model, "work", "lists/quarantine.json")
        if q:
            quarantined = set(json.load(open(q, encoding="utf-8")).get("quarantined") or [])
            row["quarantine_list"] = "present"
        else:
            row["quarantine_list"] = "missing"
    counts = Counter()
    reasons = Counter()
    ctx = contexts_info(model, {s for s, r in st.items() if "error" in r})
    for s in tg:
        r = st.get(s)
        if s in build_skip:
            b = build_skip[s]
            why = "record build: %s (n_top %s, n_contrast %s)" % (b["skip"], b.get("n_top"), b.get("n_contrast"))
            counts["no_record"] += 1
            reasons["no record: %s" % b["skip"]] += 1
            excluded.append(dict(model=L.LABEL[model], seed=s, step="no record", reason=why))
            continue
        if r is None:
            counts["no_record"] += 1
            reasons["no status row"] += 1
            excluded.append(dict(model=L.LABEL[model], seed=s, step="no record", reason="no status row"))
            continue
        if s in quarantined:
            counts["quarantined"] += 1
            excluded.append(dict(model=L.LABEL[model], seed=s, step="quarantined", reason="T2 quarantine"))
            continue
        if r.get("skip"):
            counts["skipped"] += 1
            reasons["skip: %s" % r["skip"]] += 1
            excluded.append(dict(model=L.LABEL[model], seed=s, step="skipped", reason="status skip: %s" % r["skip"]))
            continue
        if "error" in r and not r.get("ok"):
            counts["fit_error"] += 1
            c = ctx.get(s, {})
            why = "%s | contexts row: train %s, held-out strongest %s, held-out mid %s" % (
                r["error"][:120], c.get("n_train"), c.get("n_held_strong"), c.get("n_held_mid"))
            reasons["fit error: %s" % r["error"].split(":")[0]] += 1
            excluded.append(dict(model=L.LABEL[model], seed=s, step="fit error", reason=why))
            continue
        if s not in strong.index:
            counts["no_circuit"] += 1
            excluded.append(dict(model=L.LABEL[model], seed=s, step="no circuit", reason="status ok, no eval row"))
            continue
        if bool(strong.loc[s, "vac"]):
            counts["vacuous"] += 1
            excluded.append(dict(model=L.LABEL[model], seed=s, step="vacuous (strongest)",
                                 reason="merge.vacuous (v2: a_pos <= 0 or a baseline > 0.95 a_pos)"))
            continue
        counts["scored"] += 1
    extra = set(st) - set(tg)
    row.update(no_record=counts["no_record"], quarantined=counts["quarantined"], skipped=counts["skipped"],
               fit_error=counts["fit_error"], no_circuit=counts["no_circuit"], vacuous=counts["vacuous"],
               scored=counts["scored"], fitted=int(len(strong)), status_rows_off_list=len(extra),
               error_rows_total=n_err_rows, seeds_with_error_rows=len(err_seeds),
               errors_retried_to_ok=len([s for s in err_seeds if st.get(s, {}).get("ok")]),
               thin_no_mid=int(sum(1 for s in strong.index if s not in mid.index)),
               mid_vacuous=int(mid.vac.sum()) if len(mid) else 0,
               mid_scored=int((~mid.vac).sum()) if len(mid) else 0,
               reasons="; ".join("%s: %d" % kv for kv in sorted(reasons.items())))
    if len(ev):
        thin = [s for s in strong.index if s not in mid.index]
        row["thin_status_flag"] = int(sum(bool(st[s].get("thin")) for s in thin if s in st))
    assert row["targets"] == sum(row[k] for k in ("no_record", "quarantined", "skipped", "fit_error", "no_circuit",
                                                  "vacuous", "scored")), "accounting does not close for %s" % model
    return row, excluded


def main():
    rows, exc = [], []
    for model in L.MODELS:
        r, e = account(model)
        if r is None:
            continue
        rows.append(r)
        exc += e
        print("%-11s targets %d -> no record %d, quarantined %d, skipped %d, fit error %d, no circuit %d, vacuous %d"
              " -> scored %d | thin (no mid) %d, mid vacuous %d, mid scored %d | transient error rows %d on %d targets"
              " (%d retried to ok)" % (r["model"], r["targets"], r["no_record"], r["quarantined"], r["skipped"],
                                      r["fit_error"], r["no_circuit"], r["vacuous"], r["scored"], r["thin_no_mid"],
                                      r["mid_vacuous"], r["mid_scored"], r["error_rows_total"],
                                      r["seeds_with_error_rows"], r["errors_retried_to_ok"]))
        print("    reasons: %s%s" % (r["reasons"] or "-", "; frame %d eligible latents over %d components" % (
            r["frame_latents"], r["frame_components"]) if "frame_latents" in r else ""))
    if not rows:
        return
    df = pd.DataFrame(rows)
    tex = [[r["model"], L.tex_int(r.get("frame_latents")), L.tex_int(r["targets"]), L.tex_int(r["no_record"]),
            L.tex_int(r["quarantined"]), L.tex_int(r["skipped"]), L.tex_int(r["fit_error"]), L.tex_int(r["no_circuit"]),
            L.tex_int(r["vacuous"]), L.tex_int(r["scored"]), L.tex_int(r["thin_no_mid"]), L.tex_int(r["mid_vacuous"]),
            L.tex_int(r["mid_scored"])] for r in rows]
    L.write_table("exclusions", tex, ["Model", "Frame", "Targets", "No record", "Quarantined", "Skipped", "Fit error",
                                      "No circuit", "Vacuous", "Scored", "No mid pool", "Mid vacuous", "Mid scored"],
                  "lrrrrrrrrrrrr", note="Frame = eligible latents in the public target frame (fires in >= 200 "
                  "windows, <= 50%% of windows); reasons in out/run2026_exclusions.csv and run2026_excluded_targets.csv",
                  csv_df=df)
    pd.DataFrame(exc).to_csv(L.OUT / ("%sexcluded_targets.csv" % L.PREFIX), index=False)


if __name__ == "__main__":
    main()
