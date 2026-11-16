"""123 RE-SCORE of the FIXED production circuits on the rebuilt held-out sets (DAN-217). No refit.

The scorer is the production one, unchanged: 062 driver.py MODE=main with CTX_DIR = the variant's records and an OUT
that holds the production version markers (WCM_VERSION v4, WCM_R) and the production circuits (hard links, or copies
where links fail). Runner.fit loads an existing circuit instead of fitting, so nothing is refitted; prep guarantees a
circuit file for every listed target (a missing one would be FITTED by the driver, so it is never listed). The driver
resumes per target (status rows keyed by seed; eval rows keyed by seed + held) and shards with SHARD=i/k (062
launch.sh). SKIP_SPEC=1: specificity is not in the pass rule.

Variants per model (common123.VARIANTS_OF):
  orig   the production records themselves, re-scored on the SAME hardware as the new sets (the paired baseline: 115
         found H100 vs RTX pass decisions differ on 1 / 36 with median |diff| 0.03, so old and new are never mixed
         across machines)
  doc    document-disjoint (TuringLLM: original + its T5 paraphrases; public: the source document)
  topic  TuringLLM only: topic-disjoint (the 6 sibling documents of a leaf topic), sensitivity

STAGE=prep    (CPU) out/<model>/rs_<variant>/{WCM_VERSION, WCM_R, main/circuits/<key>.pt} + targets_<variant>.txt
              (targets with a circuit AND a scorable record: held-out strongest non-empty) + prep_<variant>.json
STAGE=cmd     (CPU) print the GPU launch command for every model / variant (what run_gpu.sh runs)
STAGE=verify  (CPU) after scoring: rows per variant, error rows, and the no-refit guard (every status row t_fit 0.0,
              no train.*.jsonl rows)

  python -X utf8 experiments/123-docgroup-heldout-rescore/rescore.py            (STAGE=prep)
  env: STAGE  MODELS  VARIANTS (default all of the model's)  LIMIT (first N targets per variant: smoke)
       LINK (hard | copy; default hard, falling back to copy)  + common123 path overrides
"""
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common123 as C  # noqa: E402

STAGE = os.environ.get("STAGE", "prep")
MODELS = [m for m in os.environ.get("MODELS", ",".join(C.MODELS)).split(",") if m]
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
LINK = os.environ.get("LINK", "hard")


def variants(model):
    want = [v for v in os.environ.get("VARIANTS", "").split(",") if v]
    return [v for v in C.VARIANTS_OF[model] if not want or v in want]


def link_or_copy(src, dst):
    if dst.exists():
        return "exists"
    if LINK == "hard":
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            pass
    shutil.copyfile(src, dst)
    return "copy"


def scorable(model, variant, keys):
    """(listed, excluded{key: reason}) for one variant: a target is listed when it has a production circuit file and a
    record in the variant's CTX_DIR whose held-out strongest set is non-empty (the driver's explicit split needs >= 1
    row; such a target is unscorable and counted by the report)."""
    import torch
    ctx = C.variant_ctx(model, variant)
    circ = C.source_out(model) / "main" / "circuits"
    listed, excl = [], {}
    for k in keys:
        if not (circ / ("%s.pt" % k)).exists():
            excl[k] = "no_circuit"
            continue
        p = ctx / ("%s.pt" % k)
        if not p.exists():
            excl[k] = "no_record"
            continue
        if variant != "orig":
            rec = torch.load(p, weights_only=False)[k]
            if rec.get("strong") is not None and len(rec["strong"]["held"]) == 0:
                excl[k] = "unscorable_empty_heldout"
                continue
        listed.append(k)
    return listed, excl


def prep(model):
    src = C.source_out(model)
    keys = sorted(p.stem for p in (src / "main" / "circuits").glob("*.pt"))
    for v in variants(model):
        out = C.variant_out(model, v)
        (out / "main" / "circuits").mkdir(parents=True, exist_ok=True)
        for marker in ("WCM_VERSION", "WCM_R"):
            if (src / marker).exists():
                shutil.copyfile(src / marker, out / marker)
        listed, excl = scorable(model, v, keys)
        if LIMIT:
            listed = listed[:LIMIT]
        how = {}
        for k in listed:
            r = link_or_copy(src / "main" / "circuits" / ("%s.pt" % k), out / "main" / "circuits" / ("%s.pt" % k))
            how[r] = how.get(r, 0) + 1
        tpath = C.WORK / model / ("targets_%s.txt" % v)
        tpath.write_text("\n".join(listed) + "\n")
        reasons = {}
        for r in excl.values():
            reasons[r] = reasons.get(r, 0) + 1
        info = dict(model=model, variant=v, ctx_dir=str(C.variant_ctx(model, v)), out=str(out),
                    source_out=str(src), n_circuits=len(keys), n_listed=len(listed), excluded=reasons,
                    excluded_keys=excl, circuits=how, limit=LIMIT)
        (C.WORK / model / ("prep_%s.json" % v)).write_text(json.dumps(info, indent=1))
        print("%s %-5s: %d listed of %d circuits (excluded %s; circuits %s) -> %s" % (
            model, v, len(listed), len(keys), reasons or "none", how, tpath))


def launch_cmd(model, v, gpus="0,1,2,3,4", p=6):
    """The exact launch.sh command for one model / variant, run from that model's checkout root (TL_ROOT for
    TuringLLM, PUB_ROOT for the public models: launch.sh uses relative src/ and experiments/ paths)."""
    w = "$W123"                                        # set by run_gpu.sh: this experiment's out/ on that machine
    env = ["GPUS=%s" % gpus, "P=%d" % p, "MODE=main", "ORDER=interleaved", "SKIP_SPEC=1", "WCM_VERSION=v4",
           "MERGE=0", "STAGGER=${STAGGER:-10}",
           "OUT=%s/%s/rs_%s" % (w, model, v), "TARGETS=%s/%s/targets_%s.txt" % (w, model, v),
           "LOGDIR=%s/logs/%s/rs_%s" % (w, model, v)]
    env.append("CTX_DIR=%s" % ("$CTX_ORIG_%s" % model.upper().replace("-", "_") if v == "orig"
                               else "%s/%s/ctx_%s" % (w, model, v)))
    if model != "turingllm":
        env += ["FAMILY=%s" % model, "DATA=$PUB_ROOT/data_public/%s" % model,
                "STORES=$PUB_ROOT/outputs_public/%s" % model, "SAE_PRELOAD=0"]
    return " ".join(env) + " bash experiments/062-h100-protocol-v1/launch.sh"


def verify(model):
    for v in variants(model):
        d = C.variant_out(model, v) / "main"
        st = C.read_jsonl(d / "status.jsonl")
        ev = C.read_jsonl(d / "eval.jsonl")
        tr = C.read_jsonl(d / "train.jsonl")
        tp = C.WORK / model / ("targets_%s.txt" % v)
        want = set(C.read_targets(tp)) if tp.exists() else set()
        ok = {r["seed"] for r in st if r.get("ok") or r.get("skip")}
        err = {r["seed"] for r in st if "error" in r} - ok
        fitted = sorted({r["seed"] for r in st if (r.get("t_fit") or 0) > 0})
        strong = {r["seed"] for r in ev if r.get("held") == "strong" and "error" not in r}
        print("%s %-5s: listed %d, done %d, error-only %d, strong rows %d, missing %d | REFIT GUARD: fitted %d, "
              "train rows %d -> %s" % (model, v, len(want), len(ok & want), len(err), len(strong & want),
                                       len(want - ok), len(fitted), len(tr),
                                       "OK" if not fitted and not tr else "FAIL (a circuit was fitted)"))


def main():
    for m in MODELS:
        if STAGE == "prep":
            prep(m)
        elif STAGE == "cmd":
            for v in variants(m):
                print("(cd %s && %s)" % ("$TL_ROOT" if m == "turingllm" else "$PUB_ROOT", launch_cmd(m, v)))
        elif STAGE == "verify":
            verify(m)
        else:
            raise SystemExit("STAGE must be prep, cmd or verify, got %r" % STAGE)


if __name__ == "__main__":
    main()
