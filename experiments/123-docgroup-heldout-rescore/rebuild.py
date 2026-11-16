"""123 DOCUMENT-GROUP HELD-OUT REBUILD for every production target (DAN-217). No refit: training-side fields untouched.

STAGE=plan   (CPU, every model; resumable per target)
  TuringLLM  per target and group (doc, topic): the group clause on its own (115's candidate loop, no firing, no
             duplicate rule) -> how many of the old held-out strongest rows are in a forbidden group (dropped), how
             many candidates survive (orig held + T1 scan), whether the target is certainly short (allowed < wanted) or
             certainly unscorable (0 allowed); plus the mid-band and contrast held-out leak counts (reported).
             -> out/turingllm/plan_<group>.jsonl
  public     the rebuild itself (removal only; the public path has no top-up source, like its group-aware split):
             strong.held (and, MID_RULE=drop, mid.held) lose every row whose source document supplies an activating
             training row. -> out/<model>/ctx_doc/<key>.pt + out/<model>/build_doc.jsonl
STAGE=build  (GPU, TuringLLM; SHARD=i/k; resumable per target)
  the pilot's rule, imported unchanged: 115 build_heldout.build_one (086 DAN-157 rule: firing at the anchor, exact /
  anchor-prefix / near vs training + contrast rows, no mid-pool rows, in-batch firing re-check; plus the group
  clause), filled to the original size (16) from the original held rows then the T1 scan; then MID_RULE.
  -> out/turingllm/ctx_<group>/<key>.pt + out/turingllm/build_<group>[.shard<i>].jsonl
STAGE=check  (CPU) independent re-check of every built record: 0 held rows in a forbidden group (twins included),
  no duplicate held rows, training-side fields byte-identical to the production record. -> out/<model>/check_<g>.jsonl

  python -X utf8 experiments/123-docgroup-heldout-rescore/rebuild.py                        (STAGE=plan, CPU)
  STAGE=build PYTHONPATH=src python -X utf8 experiments/123-docgroup-heldout-rescore/rebuild.py      (GPU)
  env: STAGE (plan | build | check)  MODELS (turingllm,pythia-70m,gpt2-small)  GROUPS (TuringLLM: doc,topic)
       MID_RULE (drop | keep; default drop)  LIMIT (first N targets)  SHARD (i/k, build)  TARGETS_<MODEL> (a key list,
       e.g. TARGETS_TURINGLLM; default every target with a production circuit)  + the common123 path overrides
       (TL_ROOT / PUB_ROOT / ...). The launch.sh TARGETS variable is NOT read here (it is the driver's list).
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common123 as C  # noqa: E402

STAGE = os.environ.get("STAGE", "plan")
MODELS = [m for m in os.environ.get("MODELS", ",".join(C.MODELS)).split(",") if m]
GROUPS = [g for g in os.environ.get("GROUPS", "doc,topic").split(",") if g]
MID_RULE = os.environ.get("MID_RULE", "drop")
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
SHARD_I, SHARD_K = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
if MID_RULE not in ("drop", "keep"):
    raise SystemExit("MID_RULE must be drop or keep, got %r" % MID_RULE)


def log(s):
    print("[%s 123 rebuild %s %d/%d] %s" % (time.strftime("%H:%M:%S"), STAGE, SHARD_I, SHARD_K, s), flush=True)


def production_targets(model):
    """Every target with a production main circuit file (rejected fits are saved as None and still listed: the driver
    writes their skip row), in sorted order; TARGETS_<MODEL> (e.g. TARGETS_TURINGLLM, TARGETS_GPT2_SMALL) overrides."""
    env = "TARGETS_" + model.upper().replace("-", "_")
    if os.environ.get(env):
        keys = C.read_targets(os.environ[env])
    else:
        keys = sorted(p.stem for p in (C.source_out(model) / "main" / "circuits").glob("*.pt"))
    return keys[:LIMIT] if LIMIT else keys


def load_rec(model, key):
    import torch
    p = C.source_ctx(model) / ("%s.pt" % key)
    if not p.exists():
        return None
    return torch.load(p, weights_only=False)[key]


# --------------------------------------------------------------------------------------------- plan
def plan_tl(keys):
    import torch
    sys.path.insert(0, str(C.EXP / "086-heldout-rebuild"))
    import scan as SC
    G = C.TLGroups()
    sc = SC.load_scan(C.tl_paths()["scan"])
    log("groups + scan ready (%d scan targets)" % len(sc["where"]))
    d = C.WORK / "turingllm"
    d.mkdir(parents=True, exist_ok=True)
    for g in GROUPS:
        path = d / ("plan_%s.jsonl" % g)
        done = C.done_keys(path)
        with C.open_append(path) as fh:
            for n, key in enumerate(keys):
                if key in done:
                    continue
                row = dict(seed=key, group=g, layer=int(key.split(".")[0]), kind=key.split(".")[1])
                try:
                    rec = load_rec("turingllm", key)
                    if rec is None or rec.get("strong") is None:
                        row.update(skip="no_record" if rec is None else "no_strong")
                    else:
                        gof = lambda i, _g=g: G.of(i, _g)
                        scan_ids = sc["rows"](key)["ids"].tolist() if key in sc["where"] else []
                        pl = C.tl_plan(rec, scan_ids, gof)
                        lk = C.leak_counts(rec, gof, pl["forbidden"])
                        n_want = len(rec["strong"]["held"])
                        allowed = pl["counts"]["orig_allowed"] + pl["counts"]["scan_allowed"]
                        row.update(n_want=n_want, n_old_leaky=n_want - pl["counts"]["orig_allowed"],
                                   n_forbidden_groups=len(pl["forbidden"]), cand_allowed=allowed,
                                   cand_total=pl["counts"]["orig_total"] + pl["counts"]["scan_total"],
                                   short_certain=allowed < n_want, below_floor_certain=allowed < C.FLOOR,
                                   unscorable_certain=allowed == 0, in_scan=key in sc["where"], **pl["counts"], **lk)
                except Exception as e:  # noqa: BLE001
                    row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                G.clear()
                if (n + 1) % 500 == 0:
                    log("turingllm %s %d/%d" % (g, n + 1, len(keys)))
        log("turingllm plan_%s done" % g)


def plan_pub(model, keys):
    d = C.WORK / model
    out = d / "ctx_doc"
    out.mkdir(parents=True, exist_ok=True)
    path = d / "build_doc.jsonl"
    done = C.done_keys(path)
    with C.open_append(path) as fh:
        for key in keys:
            if key in done and (out / ("%s.pt" % key)).exists():
                continue
            row = dict(seed=key, group="doc", layer=int(key.split(".")[0]), kind=key.split(".")[1])
            try:
                rec = load_rec(model, key)
                if rec is None or rec.get("strong") is None:
                    row.update(skip="no_record" if rec is None else "no_strong")
                elif "doc" not in rec["strong"]:
                    row.update(error="record has no per-row doc (069 provenance)")
                else:
                    r2, info = C.pub_rebuild(rec, MID_RULE)
                    C.save_atomic({key: r2}, out / ("%s.pt" % key))
                    row.update(info)
            except Exception as e:  # noqa: BLE001
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    log("%s build_doc done (%d targets)" % (model, len(keys)))


# --------------------------------------------------------------------------------------------- build (GPU)
def _load_115():
    spec = importlib.util.spec_from_file_location("build115", C.EXP / "115-docdisjoint-rescore-pilot" / "build_heldout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_tl(keys):
    p = C.tl_paths()
    os.environ.setdefault("SCAN", str(p["scan"]))
    os.environ.setdefault("CORPUS_DIR", str(p["corpus"]))
    B = _load_115()                                   # imports 086 rebuild (RB) and scan (SC)
    G = C.TLGroups()
    sc = B.SC.load_scan(p["scan"])
    E = B.RB.Engine()
    log("engine ready")
    mine = keys[SHARD_I::SHARD_K]
    d = C.WORK / "turingllm"
    for g in GROUPS:
        out = d / ("ctx_%s" % g)
        out.mkdir(parents=True, exist_ok=True)
        path = C.shard_path(d / ("build_%s.jsonl" % g), SHARD_I, SHARD_K)
        done = C.done_keys(d / ("build_%s.jsonl" % g))
        with C.open_append(path) as fh:
            for n, key in enumerate(mine):
                if key in done and (out / ("%s.pt" % key)).exists():
                    continue
                t = time.time()
                row = dict(seed=key, group=g)
                try:
                    rec = load_rec("turingllm", key)
                    if rec is None or rec.get("strong") is None:
                        row.update(skip="no_record" if rec is None else "no_strong")
                    elif key not in sc["where"]:
                        row.update(error="target not in the T1 scan")
                    else:
                        r2, brow = B.build_one(key, rec, sc["rows"](key), E, G, g)
                        forb = C.forbidden_groups(C.training_ids(rec), lambda i, _g=g: G.of(i, _g))
                        if MID_RULE == "drop":
                            m2, minfo = C.drop_mid(rec, lambda i, _g=g: G.of(i, _g), forb)
                            if m2 is not None:
                                r2["mid"] = m2
                        else:
                            minfo = dict(mid_rule="keep")
                        r2["docdisjoint"]["rule123"] = C.RULE123
                        r2["docdisjoint"]["mid"] = minfo
                        C.save_atomic({key: r2}, out / ("%s.pt" % key))
                        brow.pop("seed", None)
                        brow.pop("group", None)
                        row.update(brow, unscorable=brow["n_held"] == 0, **minfo)
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                row["t"] = round(time.time() - t, 2)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                G.clear()
                log("%s %d/%d %-16s %s" % (g, n + 1, len(mine), key, row.get("error") or row.get("skip") or
                                           "held %d/%d (kept %d, scan %d, old leaky %d) %.1fs" % (
                                               row["n_held"], row["n_want"], row["n_orig_kept"], row["n_scan"],
                                               row["n_old_leaky"], row["t"])))
    log("DONE")


# --------------------------------------------------------------------------------------------- check (CPU)
def check(model):
    import torch
    groups = GROUPS if model == "turingllm" else ["doc"]
    G = C.TLGroups() if model == "turingllm" else None
    for g in groups:
        d = C.WORK / model / ("ctx_%s" % g)
        rows, bad = [], 0
        for f in sorted(d.glob("*.pt")):
            key = f.stem
            new = torch.load(f, weights_only=False)[key]
            old = load_rec(model, key)
            if G is not None:
                gof = lambda i, _g=g: G.of(i, _g)
            else:                                           # public: the record's own doc per pool row
                docs = {}
                for pool in ("strong", "mid"):
                    if new.get(pool) is not None:
                        for i, dd in zip(new[pool]["ids"], new[pool]["doc"]):
                            docs[int(i)] = int(dd)
                gof = lambda i: set() if docs.get(int(i), -1) < 0 else {docs[int(i)]}
            leak, dup = C.check_disjoint(new, gof)
            s, o = new["strong"], old["strong"]
            n0 = int(o["pos"].shape[0])
            same = (s["train"] == o["train"] and new["D_train"] == old["D_train"]
                    and new.get("B_mid") == old.get("B_mid") and torch.equal(new["neg"], old["neg"])
                    and new.get("neg_ids") == old.get("neg_ids") and torch.equal(s["pos"][:n0], o["pos"])
                    and list(s["ids"][:n0]) == list(o["ids"])
                    and (old.get("mid") is None or (torch.equal(new["mid"]["pos"], old["mid"]["pos"])
                                                    and new["mid"]["train"] == old["mid"]["train"])))
            mid_leak = 0
            if new.get("mid") is not None and MID_RULE == "drop" and not new["docdisjoint"].get(
                    "mid", new["docdisjoint"]).get("mid_all_leaky_kept"):
                forb = C.forbidden_groups(C.training_ids(new), gof)
                mid_leak = sum(bool(gof(int(new["mid"]["ids"][j])) & forb) for j in new["mid"]["held"])
            ok = leak == 0 and dup == 0 and same and mid_leak == 0
            bad += not ok
            rows.append(dict(seed=key, group=g, leak=leak, dup=dup, training_unchanged=same, mid_leak=mid_leak,
                             n_held=len(s["held"]), ok=ok))
            if G is not None:
                G.clear()
        with open(C.WORK / model / ("check_%s.jsonl" % g), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        log("%s %s: %d records checked, %d FAIL" % (model, g, len(rows), bad))


def main():
    for model in MODELS:
        keys = production_targets(model)
        log("%s: %d targets" % (model, len(keys)))
        if STAGE == "plan":
            plan_tl(keys) if model == "turingllm" else plan_pub(model, keys)
        elif STAGE == "build":
            if model == "turingllm":
                build_tl(keys)
            else:
                log("%s: built in STAGE=plan (removal only, CPU)" % model)
        elif STAGE == "check":
            check(model)
        else:
            raise SystemExit("STAGE must be plan, build or check, got %r" % STAGE)


if __name__ == "__main__":
    main()
