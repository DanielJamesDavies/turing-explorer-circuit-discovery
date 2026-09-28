"""Bundle directory layout, SQLite schema and helpers shared by the build stages.

Format spec: Turing-Explorer-Back-End/v2.md (schema_version 1).
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import zlib
from typing import Any, Dict, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS latent(
    gid INTEGER PRIMARY KEY, layer INTEGER, kind TEXT, idx INTEGER,
    active_count INTEGER, frequency REAL, mean REAL, std REAL,
    seq_count INTEGER, mean_seq REAL,
    is_target INTEGER NOT NULL DEFAULT 0,
    target_status TEXT, target_reason TEXT,
    seed_cid INTEGER, n_memberships INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS target(
    gid INTEGER PRIMARY KEY, key TEXT, run TEXT, arm TEXT,
    status TEXT,              -- circuit | too_few_contexts | no_contrast | rejected
    reason TEXT, n_top INTEGER, n_mid INTEGER, n_neg INTEGER, thin INTEGER
);
CREATE TABLE IF NOT EXISTS circuit(
    cid INTEGER PRIMARY KEY, key TEXT UNIQUE, run TEXT, arm TEXT,
    seed_gid INTEGER, layer INTEGER, kind TEXT, latent INTEGER,
    topology TEXT, n_nodes INTEGER, n_edges INTEGER,
    n REAL, free0_tk REAL, freeM_topk_tk REAL, freeN_topk_tk REAL,
    phi_sup_blind_tk REAL, phi_cf_alpha_blind_tk REAL,
    amp_any INTEGER, sib_C REAL, rank_clean REAL, lifted_C REAL, near_threshold INTEGER,
    vacuous INTEGER, pass INTEGER,
    graph BLOB, meta BLOB
);
CREATE TABLE IF NOT EXISTS member(
    gid INTEGER, cid INTEGER, role TEXT, attribution REAL, alpha REAL,
    PRIMARY KEY (gid, cid)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS circuit_eval(
    cid INTEGER, held TEXT,
    n REAL, free0_tk REAL, freeM_topk_tk REAL, freeN_topk_tk REAL,
    phi_sup_blind_tk REAL, phi_cf_alpha_blind_tk REAL, phi_sup_alpha_tk REAL,
    phi_pin_alpha_topk_tk REAL, release_tk REAL, vacuous_tk INTEGER,
    free0_pre REAL, freeM_topk_pre REAL, freeN_topk_pre REAL,
    full BLOB,
    PRIMARY KEY (cid, held)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS circuit_spec(
    cid INTEGER, pi TEXT,
    rank_clean REAL, rank_circuit REAL, rank_empty REAL,
    in_topk_clean REAL, in_topk_circuit REAL, jaccard_circuit REAL,
    n_siblings_scored INTEGER, sibling_faith_median REAL,
    n_control_scored INTEGER, control_faith_median REAL,
    specificity_gap REAL, target_faith_pre REAL,
    full BLOB,
    PRIMARY KEY (cid, pi)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS train_curve(cid INTEGER PRIMARY KEY, curve BLOB);
CREATE TABLE IF NOT EXISTS target_ctx(
    gid INTEGER, pool TEXT, rank INTEGER,
    seq_id INTEGER, role TEXT, peak REAL, arg INTEGER, acts BLOB,
    PRIMARY KEY (gid, pool, rank)
) WITHOUT ROWID;
"""

# Built at finalise (stage 6), after bulk inserts.
INDEXES = """
CREATE INDEX IF NOT EXISTS circuit_seed ON circuit(seed_gid);
CREATE INDEX IF NOT EXISTS circuit_browse ON circuit(run, layer, kind);
CREATE INDEX IF NOT EXISTS member_cid ON member(cid);
"""


def clean(x: Any) -> Any:
    """JSON-safe copy: NaN/inf -> None, tensors/numpy -> python, unknown -> str."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    return str(x)


def pack(obj: Any) -> bytes:
    """zlib-compressed compact JSON (graph / meta / full-metrics blobs)."""
    return zlib.compress(json.dumps(clean(obj), separators=(",", ":"), allow_nan=False).encode(), 6)


def unpack(blob: Optional[bytes]) -> Any:
    return None if blob is None else json.loads(zlib.decompress(blob))


def num(x: Any) -> Optional[float]:
    """float or None (NaN, '', None, non-numeric)."""
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def flag(x: Any) -> Optional[int]:
    if x is None or x == "":
        return None
    if isinstance(x, str):
        return int(x.strip().lower() in ("true", "1", "yes"))
    return int(bool(x))


class Bundle:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.arrays = os.path.join(self.root, "arrays")
        self.tokens = os.path.join(self.root, "tokens")
        self.sqlite = os.path.join(self.root, "explorer.sqlite")
        self.stages_dir = os.path.join(self.root, ".stages")
        for d in (self.root, self.arrays, self.tokens, self.stages_dir):
            os.makedirs(d, exist_ok=True)

    def connect(self, build: bool = True) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite)
        if build:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA cache_size=-1000000")  # ~1 GB page cache during the build
        conn.executescript(SCHEMA)
        return conn

    # --- stage bookkeeping: each stage records what it built from, so re-runs can skip
    def _marker(self, stage: str) -> str:
        return os.path.join(self.stages_dir, f"{stage}.json")

    def stage_info(self, stage: str) -> Optional[Dict[str, Any]]:
        p = self._marker(stage)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    def mark(self, stage: str, info: Dict[str, Any]) -> None:
        info = dict(info, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self._marker(stage), "w", encoding="utf-8") as fh:
            json.dump(clean(info), fh, indent=1)

    def clear(self, stage: str) -> None:
        if os.path.exists(self._marker(stage)):
            os.remove(self._marker(stage))
