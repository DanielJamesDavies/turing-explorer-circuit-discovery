"""Experiment 080 main run: latent- vs output-targeted circuits, one method, two endpoints.

Per target (resumable at every step; one process per GPU, SHARD=i/k):
  1. contexts  the 15k run's cached record, copied read-only from SRC_OUT/ctx into OUT/ctx (as 071 does): the same
               activating contexts, split, anchors and close contrast set the LAT circuit was fitted and scored on
  2. LAT       the 062 production circuit, copied from SRC_OUT/main/circuits (never refitted), scored on both endpoints
  3. OUT arms  per arm x lambda of the run config (config_main.json, from calibrate.py's adopted recommendation):
               fit through 062's driver.Runner.fit with the output endpoint (fitting.Fitter), saved as
               OUT/circuits/<arm>/<lambda>/<key>.pt, scored on both endpoints, overlap with LAT recorded
One eval row per (target, arm, lambda) in eval.<shard>.jsonl (LAT: lambda "prod"); a failed fit or score writes an
error row and the run moves on (the row is retried on resume: done keys skip only rows without "error"). Members per
circuit in members.<shard>.jsonl.gz, training curves in train.<shard>.jsonl, one status row per target in
status.<shard>.jsonl.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -X utf8 experiments/080-latent-vs-output/run.py
  env: OUT (experiments/080-latent-vs-output/out)  TARGETS (targets.txt)  CONFIG (config_main.json)
       SRC_OUT (062 out_full/out)  ARMS (LAT,OUT_W,OUT_U; add OUT_W_FULL for the whole-model scope arm, which also
       needs a config entry)  SHARD (0/1)  LIMIT (first N targets of this shard)  ORDER (list | interleaved)
"""
import gzip
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("OUT", str(HERE / "out")))
os.environ["OUT"] = str(OUT)                                 # before fitting imports 062's driver
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import fitting as F  # noqa: E402

driver = F.driver
import torch  # noqa: E402

TARGETS = Path(os.environ.get("TARGETS", str(HERE / "targets.txt")))
CONFIG = Path(os.environ.get("CONFIG", str(HERE / "config_main.json")))
SRC_OUT = Path(os.environ.get("SRC_OUT", str(C.EXP / "062-h100-protocol-v1" / "out_full" / "out")))
ARMS = [a for a in os.environ.get("ARMS", ",".join(C.DEFAULT_ARMS)).split(",") if a]
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
ORDER = os.environ.get("ORDER", "list")
TAG = driver.TAG
for a in ARMS:
    assert a in C.ARMS, "unknown arm %r (common.ARMS)" % a


class _Stop(Exception):
    """End this target's work early (a skip record)."""


def log(s):
    print("[%s %s] %s" % (time.strftime("%H:%M:%S"), TAG, s), flush=True)


def copy_if_missing(src, dst):
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def load_meta():
    """Per-target metadata (tier, band, cell, boost_peak, lat_n) from every targets_meta*.csv (common.load_meta), so
    a run over either sampling's list finds its targets; {} when there is none."""
    return C.load_meta(HERE)


def plan(cfg):
    """[(arm, lambda, lr)] for the fitted arms requested, in the config's order."""
    out = []
    for arm in ARMS:
        if not C.ARMS[arm]["fit"]:
            continue
        if arm not in cfg["arms"]:
            raise ValueError("arm %s requested but %s has no entry for it" % (arm, CONFIG))
        a = cfg["arms"][arm]
        out += [(arm, float(l), float(a["lr"])) for l in a["lams"]]
    return out


def main():
    cfg = C.load_run_config(CONFIG)
    steps = int(cfg["steps"])
    fits = plan(cfg)
    targets = [t for t in TARGETS.read_text().split() if t][driver.SHARD_I::driver.SHARD_K]
    if ORDER == "interleaved":
        targets = driver.interleave(targets, int(os.environ.get("SLOT", "0")))
    if LIMIT:
        targets = targets[:LIMIT]
    meta = load_meta()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("run_config.%s.json" % TAG)).write_text(json.dumps(dict(cfg, config_path=str(CONFIG), arms_run=ARMS,
                                                                    targets=str(TARGETS)), indent=1))
    p = {k: OUT / ("%s.%s.jsonl" % (k, TAG)) for k in ("eval", "train", "status")}
    done = driver.jsonl_keys(p["eval"], ("seed", "arm", "lam"))
    fh = {k: open(v, "a") for k, v in p.items()}
    mh = gzip.open(OUT / ("members.%s.jsonl.gz" % TAG), "at")
    log("%d targets | arms %s | %d fits per target (%s) | steps %d | logit_norm %s | out %s" % (
        len(targets), ",".join(ARMS), len(fits), " ".join("%s@%g/lr%g" % f for f in fits), steps,
        cfg["logit_norm"], OUT))
    Fi = F.Fitter(cfg)
    R = Fi.R
    for n_, key in enumerate(targets):
        want = ([(key, "LAT", "prod")] if "LAT" in ARMS else []) + [(key, a, C.lam_tag(l)) for a, l, _ in fits]
        if all(k in done for k in want):
            continue
        m = meta.get(key, {})
        base = dict(seed=key, tier=m.get("tier"), band=m.get("band") or C.band(C.parse(key)[0]),
                    cell=m.get("cell") or None, boost_peak=float(m["boost_peak"]) if m.get("boost_peak") else None,
                    shard=TAG)
        st = dict(base, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
        t_all = time.time()
        try:
            t = time.time()
            copy_if_missing(SRC_OUT / "ctx" / ("%s.pt" % key), OUT / "ctx" / ("%s.pt" % key))
            rec = R.contexts(key)
            st["t_ctx"] = round(time.time() - t, 1)
            if rec.get("strong") is None:
                st["skip"] = "no_contrast" if rec.get("no_contrast") else "too_few_contexts"
                raise _Stop
            tr_arm, thin = R.train_arm(rec)
            base.update(train_arm=tr_arm, thin=thin)
            lat_path = OUT / "circuits" / "LAT" / ("%s.pt" % key)
            copy_if_missing(SRC_OUT / "main" / "circuits" / ("%s.pt" % key), lat_path)
            lat = torch.load(lat_path, weights_only=False) if lat_path.exists() else None
            lat_m = F.member_set(lat)
            st["lat_n"] = len(lat_m) if lat is not None else None
            if "LAT" in ARMS and (key, "LAT", "prod") not in done:
                row = dict(base, arm="LAT", lam="prod", lam_value=C.PROD_LAM, lr=C.PROD_LR, steps=C.PROD_STEPS,
                           weighted=True, scope="upstream")
                try:
                    if lat is None:
                        row["skip"] = "no_lat_circuit"
                    else:
                        row.update(F.scalars(Fi.score(lat, rec, "upstream")), n_nodes=driver.n_nodes(lat))
                        row.update(arm="LAT", lam="prod")
                except Exception as e:  # noqa: BLE001
                    row["error"] = "score %s: %s" % (type(e).__name__, str(e)[:300])
                    traceback.print_exc()
                fh["eval"].write(json.dumps(row) + "\n"); fh["eval"].flush()
                mh.write(json.dumps(dict(seed=key, arm="LAT", lam="prod", members=sorted(lat_m))) + "\n"); mh.flush()
            for arm, lam, lr in fits:
                lt = C.lam_tag(lam)
                if (key, arm, lt) in done:
                    continue
                a = C.ARMS[arm]
                row = dict(base, arm=arm, lam=lt, lam_value=lam, lr=lr, steps=steps, weighted=bool(a["weighted"]),
                           scope=a["scope"], logit_norm=cfg["logit_norm"])
                path = OUT / "circuits" / arm / lt / ("%s.pt" % key)
                fresh = not path.exists()
                try:
                    c, t_fit, peak = Fi.fit(arm, lam, lr, steps, rec, key, path, train_fh=fh["train"],
                                            label=dict(arm=arm, lam=lt, lr=lr, steps=steps))
                    row.update(t_fit=round(t_fit, 1), peak_mb_fit=peak if fresh else None, fresh_fit=fresh)
                    if c is None:
                        row["skip"] = "rejected"
                    else:
                        mem = F.member_set(c)
                        ov = C.overlap(lat_m, mem) if lat is not None else {}
                        t = time.time()
                        sc = Fi.score(c, rec, str(a["scope"]),
                                      extra_scopes=("upstream",) if a["scope"] != "upstream" else ())
                        row.update(F.scalars(sc), n_nodes=driver.n_nodes(c), t_score=round(time.time() - t, 1),
                                   jaccard_lat=ov.get("jaccard"), cont_lat_in_arm=ov.get("cont_a_in_b"),
                                   cont_arm_in_lat=ov.get("cont_b_in_a"), n_inter_lat=ov.get("n_inter"))
                        row.update(arm=arm, lam=lt)
                        mh.write(json.dumps(dict(seed=key, arm=arm, lam=lt, members=sorted(mem))) + "\n"); mh.flush()
                except Exception as e:  # noqa: BLE001
                    row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                    traceback.print_exc()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                fh["eval"].write(json.dumps(row) + "\n"); fh["eval"].flush()
                log("%-16s %-6s lam %-7s n=%-6s fit %6.1fs  Z %s oZ %s J %s %s" % (
                    key, arm, lt, row.get("n"), row.get("t_fit") or 0, fmt(row.get("free0_tk")),
                    fmt(row.get("out_free0")), fmt(row.get("jaccard_lat")), row.get("error") or row.get("skip") or ""))
            st["ok"] = True
        except _Stop:
            pass
        except Exception as e:  # noqa: BLE001
            st["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        st["t_total"] = round(time.time() - t_all, 1)
        fh["status"].write(json.dumps(st) + "\n"); fh["status"].flush()
        log("%4d/%d %-16s %s total %ss" % (n_ + 1, len(targets), key, st.get("error") or st.get("skip") or "ok",
                                          st["t_total"]))
    mh.close()
    log("DONE")


def fmt(v):
    return "%.2f" % v if isinstance(v, (int, float)) and v == v else "-"


if __name__ == "__main__":
    main()
