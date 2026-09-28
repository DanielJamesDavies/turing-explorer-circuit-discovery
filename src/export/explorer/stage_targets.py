"""Stage 3: target table + the run's own context pools with roles (target_ctx).

Pools (rank = row order in the run's ctx file):
  strong  top contexts        role train | held_strong | unused
  mid     mid-band contexts   role train | held_mid    | unused
  neg     contrast contexts   role contrast_train | contrast_held (target verified silent; no peak/arg)
'train' means the context was in the fit's training set (contexts.jsonl train_ids); 'unused'
means it sits in the training split but was not selected (arm B trains on 32 strong + 16 mid).
Skipped targets were never fitted, so their contexts only get held/unused roles.
"""
from __future__ import annotations

import time

from export.explorer import ids
from export.explorer.bundle import Bundle
from export.explorer.run_source import RunSource


def _pool_rows(gid, pool, p, train_ids, held_role):
    held = {int(j) for j in p["held"]}
    rows = []
    for rank, sid in enumerate(p["ids"]):
        sid = int(sid)
        role = "train" if sid in train_ids else held_role if rank in held else "unused"
        rows.append((gid, pool, rank, sid, role, float(p["peak"][rank]), int(p["arg"][rank]), None))
    return rows


def build(bundle: Bundle, run: RunSource, run_name: str) -> dict:
    t0 = time.time()
    conn = bundle.connect()
    conn.execute("DELETE FROM target_ctx")
    conn.execute("DELETE FROM target")
    n_rows, n_targets = 0, 0
    for key in run.targets:
        layer, kind, latent = ids.parse_key(key)
        gid = ids.gid_of(layer, kind, latent)
        rec = run.ctx(key)
        status, reason = run.target_status(key)
        crow = run.contexts.get(key, {})
        train_ids = {int(s) for s in crow.get("train_ids", [])}
        rows = []
        n_neg = 0
        if rec is not None:
            if rec.get("strong") is not None:
                rows += _pool_rows(gid, "strong", rec["strong"], train_ids, "held_strong")
            if rec.get("mid") is not None:
                rows += _pool_rows(gid, "mid", rec["mid"], train_ids, "held_mid")
            neg_ids = rec.get("neg_ids")
            if neg_ids is not None:
                n_neg = len(neg_ids)
                if crow:
                    held_neg = {int(s) for s in crow.get("contrast_held_ids", [])}
                else:  # engine split rule: first n - round(n/4) train, rest held out
                    held_neg = {int(s) for s in neg_ids[n_neg - round(n_neg / 4):]}
                rows += [(gid, "neg", r, int(s), "contrast_held" if int(s) in held_neg else "contrast_train",
                          None, None, None) for r, s in enumerate(neg_ids)]
        conn.executemany("INSERT INTO target_ctx VALUES (?,?,?,?,?,?,?,?)", rows)
        st = run.status.get(key, {})
        conn.execute("INSERT INTO target VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (gid, key, run_name, crow.get("arm") or st.get("arm"), status, reason,
                      (rec or {}).get("n_top"), (rec or {}).get("n_mid"), n_neg,
                      int(bool(st.get("thin"))) if "thin" in st else int((rec or {}).get("mid") is None)))
        n_rows += len(rows)
        n_targets += 1
        if n_targets % 500 == 0:
            print(f"    targets: {n_targets}/{len(run.targets)} ({time.time() - t0:.0f}s)", flush=True)
    conn.commit()
    conn.close()
    return dict(run=run.dir, run_name=run_name, n_targets=n_targets, n_ctx_rows=n_rows, secs=round(time.time() - t0, 1))
