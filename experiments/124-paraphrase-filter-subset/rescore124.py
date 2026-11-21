"""124 RE-SCORE prep / verify (CPU) for the fixed production circuits on the subset, two variants (DAN-217):
  orig  the production records (T2 ctx_fresh) re-scored on the local GPU: the paired baseline (115: H100 vs RTX pass
        decisions differ on 1/36, so the old sets are re-scored on the same machine as the new ones)
  para  the paraphrase-filtered records (out/ctx_para, rebuild124 STAGE=build on the GPU)
The scorer is the production one, unchanged (062 driver.py MODE=main, as 115 / 123 run it): an OUT holding the
production markers (WCM_VERSION v4, WCM_R) and hard links (copies where links fail) to the production circuits, so
Runner.fit loads them and nothing is refitted. A target without a circuit is never listed (the driver would fit it).

STAGE=prep    out/rs_<v>/{WCM_VERSION, WCM_R, main/circuits/<key>.pt}, out/targets_<v>.txt, out/prep_<v>.json
              (para lists the subset targets whose rebuilt held-out set is non-empty; the rest are 'unscorable')
STAGE=cmd     print the local driver commands (what run_local.sh runs)
STAGE=verify  rows per variant, error rows, and the refit guard (every status row t_fit 0, no train rows)

  python -X utf8 experiments/124-paraphrase-filter-subset/rescore124.py          (STAGE=prep)
  env: STAGE  VARIANTS (orig,para)  LINK (hard | copy)
"""
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import para124 as P  # noqa: E402

C = P.C
STAGE = os.environ.get("STAGE", "prep")
VARIANTS = [v for v in os.environ.get("VARIANTS", "orig,para").split(",") if v]
LINK = os.environ.get("LINK", "hard")


def ctx_of(v):
    return C.tl_paths()["ctx"] if v == "orig" else P.WORK / ("ctx_%s" % v)


def out_of(v):
    return P.WORK / ("rs_%s" % v)


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


def scorable(v, keys):
    import torch
    circ = C.tl_paths()["out"] / "main" / "circuits"
    listed, excl = [], {}
    for k in keys:
        if not (circ / ("%s.pt" % k)).exists():
            excl[k] = "no_circuit"
            continue
        p = ctx_of(v) / ("%s.pt" % k)
        if not p.exists():
            excl[k] = "no_record"
            continue
        if v != "orig" and len(torch.load(p, weights_only=False)[k]["strong"]["held"]) == 0:
            excl[k] = "unscorable_empty_heldout"
            continue
        listed.append(k)
    return listed, excl


def prep():
    src = C.tl_paths()["out"]
    keys = C.read_targets(P.subset_path())
    for v in VARIANTS:
        out = out_of(v)
        (out / "main" / "circuits").mkdir(parents=True, exist_ok=True)
        for marker in ("WCM_VERSION", "WCM_R"):
            if (src / marker).exists():
                shutil.copyfile(src / marker, out / marker)
        listed, excl = scorable(v, keys)
        how = {}
        for k in listed:
            r = link_or_copy(src / "main" / "circuits" / ("%s.pt" % k), out / "main" / "circuits" / ("%s.pt" % k))
            how[r] = how.get(r, 0) + 1
        (P.WORK / ("targets_%s.txt" % v)).write_text("\n".join(listed) + "\n")
        reasons = {}
        for r in excl.values():
            reasons[r] = reasons.get(r, 0) + 1
        P.write_json(dict(variant=v, ctx_dir=str(ctx_of(v)), out=str(out), source_out=str(src), n_subset=len(keys),
                          n_listed=len(listed), excluded=reasons, excluded_keys=excl, circuits=how),
                     P.WORK / ("prep_%s.json" % v))
        print("%-5s: %d listed of %d (excluded %s; circuits %s)" % (v, len(listed), len(keys), reasons or "none", how))


def driver_cmd(v, rel=True):
    """The 062 driver command for one variant (single process, as the 115 pilot ran it), from the repo root in WSL."""
    e = "experiments/124-paraphrase-filter-subset/out"
    ctx = C.tl_paths()["ctx"].relative_to(C.REPO).as_posix() if v == "orig" else "%s/ctx_%s" % (e, v)
    return ("MODE=main ORDER=list SKIP_SPEC=1 WCM_VERSION=v4 OUT=%s/rs_%s CTX_DIR=%s TARGETS=%s/targets_%s.txt "
            "PYTHONPATH=src .venv/bin/python -X utf8 experiments/062-h100-protocol-v1/driver.py" % (e, v, ctx, e, v))


def verify():
    bad = 0
    for v in VARIANTS:
        d = out_of(v) / "main"
        st, ev, tr = (C.read_jsonl(d / ("%s.jsonl" % s)) for s in ("status", "eval", "train"))
        tp = P.WORK / ("targets_%s.txt" % v)
        want = set(C.read_targets(tp)) if tp.exists() else set()
        ok = {r["seed"] for r in st if r.get("ok") or r.get("skip")}
        err = {r["seed"] for r in st if "error" in r} - ok
        fitted = sorted({r["seed"] for r in st if (r.get("t_fit") or 0) > 0})
        strong = {r["seed"] for r in ev if r.get("held") == "strong" and "error" not in r}
        guard = not fitted and not tr
        bad += (not guard) or bool(want - ok)
        print("%-5s: listed %d, done %d, error-only %d, strong rows %d, missing %d | REFIT GUARD: fitted %d, train rows "
              "%d -> %s" % (v, len(want), len(ok & want), len(err), len(strong & want), len(want - ok), len(fitted),
                            len(tr), "OK" if guard else "FAIL (a circuit was fitted)"))
    return bad


def main():
    if STAGE == "prep":
        prep()
    elif STAGE == "cmd":
        for v in VARIANTS:
            print(driver_cmd(v))
    elif STAGE == "verify":
        sys.exit(1 if verify() else 0)
    else:
        raise SystemExit("STAGE must be prep, cmd or verify")


if __name__ == "__main__":
    main()
