"""124 PARAPHRASE-FILTERED HELD-OUT SETS for a stratified TuringLLM subset (DAN-217). No refit: training-side fields
are the production record's, byte for byte. The rule is in para124's docstring.

STAGE=idf    (CPU) fit 114's IDF on 114's exact 100k-window sample -> out/idf114.npz, then VERIFY it: recompute the
             max-cos of 114's held rows (out/rows.jsonl, VERIFY_N records) from the records and compare with 114's
             stored cos (must agree to float32 precision, and the >= 0.3 calls must be identical).
STAGE=plan   (CPU) per subset target, the paraphrase test alone (before firing / the 086 duplicate rule): old held rows
             dropped, candidates allowed (orig held + T1 scan), certainly short / below floor / unscorable, mid-band
             held rows dropped, contrast held rows that are paraphrases (measured). -> out/plan_para.jsonl
STAGE=build  the rebuild: 115 build_heldout.build_one (086 DAN-157 rule + the clause, here the paraphrase test),
             fill to the original size from the original held rows then the T1 scan; MID_RULE drop|keep.
             DEVICE=cuda (default; the real records, out/ctx_para/ + out/build_para.jsonl) or DEVICE=cpu (the dry run:
             the same code with the protocol forward on the CPU, fp32 instead of the GPU's bf16 autocast; writes
             out/ctx_para_cpu/ + out/build_para_cpu.jsonl, never read by the scorer).
STAGE=check  (CPU) every built record: 0 held rows with cos >= 0.3 to a training row, no duplicate held rows, training
             side byte-identical, mid held rows clean. -> out/check_para[_cpu].jsonl

  python -X utf8 experiments/124-paraphrase-filter-subset/rebuild124.py                     (STAGE=idf)
  wsl: STAGE=build PYTHONPATH=src .venv/bin/python -X utf8 experiments/124-paraphrase-filter-subset/rebuild124.py
  env: STAGE  DEVICE (cuda | cpu)  MID_RULE (drop | keep)  LIMIT  SUBSET (targets_subset.txt)  VERIFY_N (60)
       + the common123 path overrides (TL_ROOT ...) and TL_DATA (the corpus shards, default data/)
Every stage resumes per target (done = a complete jsonl row, and for build the record file).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import para124 as P  # noqa: E402

C = P.C
sys.path.insert(0, str(C.REPO / "src"))
STAGE = os.environ.get("STAGE", "idf")
DEVICE = os.environ.get("DEVICE", "cuda")
MID_RULE = os.environ.get("MID_RULE", "drop")
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
SUF = "" if DEVICE == "cuda" else "_cpu"
if MID_RULE not in ("drop", "keep") or DEVICE not in ("cuda", "cpu"):
    raise SystemExit("MID_RULE must be drop|keep and DEVICE cuda|cpu")


def log(s):
    print("[%s 124 %s%s] %s" % (time.strftime("%H:%M:%S"), STAGE, SUF, s), flush=True)


def keys():
    k = C.read_targets(P.subset_path())
    return k[:LIMIT] if LIMIT else k


def load_rec(key):
    import torch
    p = C.tl_paths()["ctx"] / ("%s.pt" % key)
    return torch.load(p, weights_only=False)[key] if p.exists() else None


# --------------------------------------------------------------------------------------------- idf
def stage_idf():
    src = np.load(C.tl_paths()["corpus"] / "corpus_src.npy", mmap_mode="r")
    n_win = len(src) - 1
    if n_win != P.N_WIN_114:
        raise SystemExit("corpus has %d windows, 114 had %d: not the same sample" % (n_win, P.N_WIN_114))
    ids = P.idf_sample_ids(n_win)
    t = time.time()
    rows = P.Corpus().windows(ids)
    log("read %d IDF windows in %.0fs" % (len(rows), time.time() - t))
    tf = P.K114.Tfidf(rows)
    P.WORK.mkdir(parents=True, exist_ok=True)
    np.savez(P.idf_path(), idf=tf.idf, n=tf.n, ids_md5=P.ids_md5(ids), seed=P.IDF_SEED, n_win=n_win)
    log("idf -> %s (ids md5 %s)" % (P.idf_path(), P.ids_md5(ids)))
    verify_idf(P.load_tfidf())


def verify_idf(tf, n_rec=None, write=True):
    """Recompute 114's per-held-row max cos (same records, same training rows) and compare with out/rows.jsonl."""
    n_rec = n_rec or int(os.environ.get("VERIFY_N", "60"))
    by = {}
    for line in open(C.EXP / "114-paraphrase-check" / "out" / "rows.jsonl", encoding="utf-8"):
        r = json.loads(line)
        by.setdefault(r["key"], []).append(r)
    ks = sorted(by)[:n_rec]
    d, calls, n = [], 0, 0
    for k in ks:
        rec = load_rec(k)
        s = rec["strong"]
        cm = P.ParaRule(tf, P.train_rows(rec)).cos_max(np.asarray(s["pos"][s["held"]]))
        for r in by[k]:
            d.append(abs(float(cm[r["row"]]) - r["cos"]))
            calls += (cm[r["row"]] >= 0.3) != (r["cos"] >= 0.3)
            n += 1
    out = dict(records=len(ks), rows=n, max_abs_diff=float(max(d)), median_abs_diff=float(np.median(d)),
               call_mismatches=int(calls), ok=bool(max(d) < 1e-5 and calls == 0))
    if write:
        P.write_json(out, P.WORK / "idf_verify.json")
    log("verify vs 114 rows.jsonl: %s" % out)
    if not out["ok"]:
        raise SystemExit("IDF does not reproduce 114")
    return out


# --------------------------------------------------------------------------------------------- plan
def stage_plan():
    sys.path.insert(0, str(C.EXP / "086-heldout-rebuild"))
    import scan as SC
    tf, corpus = P.load_tfidf(), P.Corpus()
    sc = SC.load_scan(C.tl_paths()["scan"])
    path = P.WORK / "plan_para.jsonl"
    done = C.done_keys(path)
    ks = keys()
    with C.open_append(path) as fh:
        for n, key in enumerate(ks):
            if key in done:
                continue
            row = dict(seed=key, group=P.GROUP, layer=int(key.split(".")[0]), kind=key.split(".")[1])
            try:
                rec = load_rec(key)
                G = P.ParaGroups(rec, tf, corpus)
                scan_ids = sc["rows"](key)["ids"].tolist()
                row.update(P.plan_one(rec, scan_ids, G, tf))
            except Exception as e:  # noqa: BLE001
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if (n + 1) % 50 == 0:
                log("%d/%d" % (n + 1, len(ks)))
    log("plan done (%d targets)" % len(ks))


# --------------------------------------------------------------------------------------------- build
def make_engine(B):
    if DEVICE == "cuda":
        return B.RB.Engine()
    import torch

    class CPUEngine(B.RB.Engine):
        def __init__(self):                                   # 086 Engine.__init__ with the device set to cpu
            from data.loader import DataLoader
            from model.inference import Inference
            from sae.bank import SAEBank
            self.dev = torch.device("cpu")
            self.loader = DataLoader(device=self.dev)
            self.inference = Inference(device=self.dev)
            self.bank = SAEBank(device=self.dev, load_decoders=False)
            self.kinds = list(self.bank.kinds)
            self._sel = None
    torch.set_num_threads(int(os.environ.get("CPU_THREADS", "12")))
    return CPUEngine()


def stage_build():
    import importlib.util
    p = C.tl_paths()
    os.environ.setdefault("SCAN", str(p["scan"]))
    os.environ.setdefault("CORPUS_DIR", str(p["corpus"]))
    spec = importlib.util.spec_from_file_location("build115", C.EXP / "115-docdisjoint-rescore-pilot" / "build_heldout.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    tf, corpus = P.load_tfidf(), P.Corpus()
    sc = B.SC.load_scan(p["scan"])
    t0 = time.time()
    E = make_engine(B)
    log("engine ready on %s (%.0fs)" % (DEVICE, time.time() - t0))
    out = P.WORK / ("ctx_%s%s" % (P.GROUP, SUF))
    out.mkdir(parents=True, exist_ok=True)
    path = P.WORK / ("build_%s%s.jsonl" % (P.GROUP, SUF))
    done = C.done_keys(path)
    ks = keys()
    for n, key in enumerate(ks):
        if key in done and (out / ("%s.pt" % key)).exists():
            continue
        t = time.time()
        row = dict(seed=key, group=P.GROUP, device=DEVICE)
        try:
            rec = load_rec(key)
            rows = sc["rows"](key)
            G = P.ParaGroups(rec, tf, corpus)
            s = rec["strong"]
            G.prefetch([int(s["ids"][j]) for j in s["held"]] + [int(x) for x in rows["ids"].tolist()])
            r2, brow = B.build_one(key, rec, rows, E, G, P.GROUP)
            if MID_RULE == "drop":
                m2, minfo = C.drop_mid(rec, G.of, P.FORBIDDEN)
                if m2 is not None:
                    r2["mid"] = m2
            else:
                minfo = dict(mid_rule="keep")
            held_cos = [G.cos.get(int(r2["strong"]["ids"][j])) for j in r2["strong"]["held"]]
            r2["docdisjoint"].update(rule124=P.RULE124, thresh=P.THRESH, mid=minfo, held_cos=held_cos,
                                     idf=str(P.idf_path().name))
            C.save_atomic({key: r2}, out / ("%s.pt" % key))
            for f in ("seed", "group", "new_share_contrast", "old_share_contrast"):   # group-clause fields: not
                brow.pop(f, None)                                                     # meaningful for this rule
            row.update(brow, unscorable=brow["n_held"] == 0,
                       new_cos_max=max([c for c in held_cos if c is not None], default=None), **minfo)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        row["t"] = round(time.time() - t, 2)
        with C.open_append(path) as fh:
            fh.write(json.dumps(row) + "\n")
        log("%d/%d %-16s %s" % (n + 1, len(ks), key, row.get("error") or
                                "held %d/%d (kept %d, scan %d, old para %d) %.1fs" % (
                                    row["n_held"], row["n_want"], row["n_orig_kept"], row["n_scan"],
                                    row["n_old_leaky"], row["t"])))
    log("DONE")


# --------------------------------------------------------------------------------------------- check
def stage_check():
    import torch
    tf, corpus = P.load_tfidf(), P.Corpus()
    d = P.WORK / ("ctx_%s%s" % (P.GROUP, SUF))
    rows, bad = [], 0
    for f in sorted(d.glob("*.pt")):
        key = f.stem
        new = torch.load(f, weights_only=False)[key]
        old = load_rec(key)
        G = P.ParaGroups(new, tf, corpus)                    # training rows are the old ones (checked below)
        leak, dup = C.check_disjoint(new, G.of)
        same = P.training_unchanged(new, old)
        mid_leak = 0
        if new.get("mid") is not None and MID_RULE == "drop" and not new["docdisjoint"]["mid"].get("mid_all_leaky_kept"):
            mid_leak = sum(bool(G.of(int(new["mid"]["ids"][j])) & P.FORBIDDEN) for j in new["mid"]["held"])
        ok = leak == 0 and dup == 0 and same and mid_leak == 0
        bad += not ok
        rows.append(dict(seed=key, leak=leak, dup=dup, training_unchanged=same, mid_leak=mid_leak,
                         n_held=len(new["strong"]["held"]), ok=ok))
    with open(P.WORK / ("check_%s%s.jsonl" % (P.GROUP, SUF)), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    log("%d records checked, %d FAIL" % (len(rows), bad))
    if bad:
        raise SystemExit(1)


def main():
    {"idf": stage_idf, "verify": lambda: verify_idf(P.load_tfidf()), "plan": stage_plan, "build": stage_build,
     "check": stage_check}[STAGE]()


if __name__ == "__main__":
    main()
