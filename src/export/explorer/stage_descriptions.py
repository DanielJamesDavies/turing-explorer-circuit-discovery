"""Stage 9: auto-generated circuit descriptions (DAN-154) -> bundle/descriptions.sqlite.

Source: experiments/065-circuit-describer/results/descriptions.jsonl, written only by that experiment's
combine.py (one record per target key, the latest successful description). This stage only reads it. Each
description was written by Claude Sonnet from the circuit's reading data (065 describe.py): a label, a short
description, and two facets, `trigger` (what the target fires on) and `condition` (what else it needs). They are
hypotheses, not verified.

Coverage is partial (the describer runs layer by layer): a circuit has a description iff it has a
circuit_description row. Records whose `result` is null (failed calls) are skipped. The file is rebuilt from the
jsonl on every run, written to a temporary file and swapped in with an atomic replace, so a reader never sees a
half-written file; the back end opens it per request, so the swap only waits for in-flight reads.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import Counter
from typing import Optional

from export.explorer import ids
from export.explorer.bundle import Bundle

DESCRIPTIONS_FILE = "descriptions.sqlite"
SOURCE = os.path.join("experiments", "065-circuit-describer", "results", "descriptions.jsonl")

# 065 describe.py TRIGGERS / CONDITIONS, shortened (the UI shows them as tooltips).
TRIGGERS = {
    "one_token": "one token, with its case / plural / prefix variants",
    "word_family": "several different words or pieces sharing a meaning or role",
    "varied": "no dominant token: it peaks on varied or function words across a passage",
    "number_format": "digits, dates, numbers, list or heading structure",
    "unclear": "the peaks do not fit any of these",
}
CONDITIONS = {
    "none": "fires on the trigger wherever it appears",
    "within_word": "only inside a particular word, or after a particular word piece",
    "construction": "only in a grammatical construction or slot",
    "subject_area": "only within one subject area",
    "entity_kind": "only after a certain kind of name or entity",
    "unclear": "the evidence does not say",
}

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE circuit_description(
    cid INTEGER PRIMARY KEY, seed_gid INTEGER UNIQUE, key TEXT, circuit_key TEXT,
    label TEXT, description TEXT, trigger TEXT, condition TEXT,
    trigger_computed INTEGER, merged_consistency REAL,
    model_id TEXT, prompt_version INTEGER, time TEXT
);
CREATE INDEX circuit_description_trigger ON circuit_description(trigger);
CREATE INDEX circuit_description_condition ON circuit_description(condition);
CREATE VIRTUAL TABLE description_fts USING fts5(
    label, description, content='circuit_description', content_rowid='cid',
    tokenize='unicode61 remove_diacritics 2'
);
"""


def descriptions_path(bundle: Bundle) -> str:
    return os.path.join(bundle.root, DESCRIPTIONS_FILE)


def _js(x) -> str:
    return json.dumps(x, separators=(",", ":"))


def _replace(tmp: str, path: str, tries: int = 8) -> None:
    """os.replace, retried: on Windows it fails while a reader has the old file open."""
    for t in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if t == tries - 1:
                raise
            time.sleep(0.25 * (t + 1))


# ---------------------------------------------------------------------------------------------- coverage + checks
def coverage(bundle: Bundle, main=None) -> Optional[dict]:
    """Coverage summary of descriptions.sqlite (None when it does not exist)."""
    path = descriptions_path(bundle)
    if not os.path.isfile(path):
        return None
    own = main is None
    if own:
        main = sqlite3.connect(f"file:{bundle.sqlite}?mode=ro", uri=True)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    total = dict(main.execute("SELECT layer, COUNT(*) FROM circuit GROUP BY layer").fetchall())
    have = {}
    for (cid,) in conn.execute("SELECT cid FROM circuit_description"):
        row = main.execute("SELECT layer FROM circuit WHERE cid = ?", (cid,)).fetchone()
        if row is not None:
            have[row[0]] = have.get(row[0], 0) + 1
    meta = {k: json.loads(v) for k, v in conn.execute("SELECT key, value FROM meta")}
    out = dict(file=DESCRIPTIONS_FILE, circuits=sum(have.values()), circuits_total=sum(total.values()),
               by_layer={str(l): [have.get(l, 0), total[l]] for l in sorted(total)},
               triggers=dict(conn.execute("SELECT trigger, COUNT(*) FROM circuit_description GROUP BY trigger")),
               conditions=dict(conn.execute("SELECT condition, COUNT(*) FROM circuit_description GROUP BY condition")),
               models=dict(conn.execute("SELECT model_id, COUNT(*) FROM circuit_description GROUP BY model_id")),
               prompt_versions={str(k): v for k, v in conn.execute(
                   "SELECT prompt_version, COUNT(*) FROM circuit_description GROUP BY prompt_version")},
               source=meta.get("source"), source_records=meta.get("n_records"),
               skipped_failed=meta.get("n_failed"), unmatched_keys=meta.get("n_unmatched"))
    conn.close()
    if own:
        main.close()
    return out


def validate(bundle: Bundle, main, ck) -> None:
    """Consistency checks that hold at any coverage (stage 6 and the end of stage 9). ck(name, ok, detail)."""
    path = descriptions_path(bundle)
    if not os.path.isfile(path):
        return
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    q = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731
    rows = conn.execute("SELECT cid, seed_gid, key, circuit_key FROM circuit_description").fetchall()
    bad = []
    for cid, seed, key, ckey in rows:
        c = main.execute("SELECT seed_gid, key FROM circuit WHERE cid = ?", (cid,)).fetchone()
        if c is None or c != (seed, ckey) or ids.gid_of(*ids.parse_key(key)) != seed:
            bad.append(key)
    ck("description keys map to their circuits (cid, seed, key)", not bad, bad[:5] or f"{len(rows)} described")
    n_unmatched = json.loads(conn.execute("SELECT value FROM meta WHERE key = 'n_unmatched'").fetchone()[0])
    ck("every described key in the source names a circuit", n_unmatched == 0, n_unmatched)
    bad_t = [r[0] for r in conn.execute("SELECT DISTINCT trigger FROM circuit_description") if r[0] not in TRIGGERS]
    ck("description triggers in the allowed set", not bad_t, bad_t or "ok")
    bad_c = [r[0] for r in conn.execute("SELECT DISTINCT condition FROM circuit_description") if r[0] not in CONDITIONS]
    ck("description conditions in the allowed set", not bad_c, bad_c or "ok")
    ck("descriptions have a label and text",
       q("SELECT COUNT(*) FROM circuit_description WHERE COALESCE(label, '') = '' OR COALESCE(description, '') = ''") == 0)
    ck("description search index matches the rows",
       q("SELECT COUNT(*) FROM description_fts") == len(rows), len(rows))
    conn.close()


def _update_manifest(bundle: Bundle, cov: dict, info: dict) -> None:
    path = os.path.join(bundle.root, "manifest.json")
    if not os.path.isfile(path):
        print("  no manifest.json yet; stage 6 will record the descriptions", flush=True)
        return
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest.setdefault("features", {})["has_descriptions"] = bool(cov["circuits"])
    manifest["descriptions"] = cov
    manifest.setdefault("stages", {})["9"] = info
    manifest.setdefault("sizes", {})[DESCRIPTIONS_FILE] = os.path.getsize(descriptions_path(bundle))
    if manifest.get("checksums"):                       # stage 6 wrote checksums: keep this file's current
        from export.explorer.stage_finalize import _sha256
        manifest["checksums"][DESCRIPTIONS_FILE] = _sha256(descriptions_path(bundle))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, path)
    print("  manifest.json: features.has_descriptions, descriptions section updated", flush=True)


# ---------------------------------------------------------------------------------------------- build
def build(bundle: Bundle, repo_root: str, source: str = "") -> dict:
    t0 = time.time()
    source = source or os.path.join(repo_root, SOURCE)
    main = sqlite3.connect(f"file:{bundle.sqlite}?mode=ro", uri=True)
    circuits = {seed: (cid, key) for cid, seed, key in main.execute(
        "SELECT c.cid, c.seed_gid, c.key FROM circuit c JOIN target t ON t.gid = c.seed_gid AND t.run = c.run")}

    rows, n_rec, n_failed, unmatched, bad_facet = [], 0, 0, [], []
    with open(source, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_rec += 1
            res = rec.get("result")
            if not res:
                n_failed += 1
                continue
            try:
                seed = ids.gid_of(*ids.parse_key(rec["key"]))
            except (ValueError, KeyError):
                unmatched.append(rec.get("key"))
                continue
            if seed not in circuits:
                unmatched.append(rec["key"])
                continue
            if res.get("trigger") not in TRIGGERS or res.get("condition") not in CONDITIONS:
                bad_facet.append(rec["key"])
                continue
            cid, ckey = circuits[seed]
            rows.append((cid, seed, rec["key"], ckey, res["label"].strip(), res["description"].strip(),
                         res["trigger"], res["condition"],
                         None if rec.get("trigger_computed") is None else int(bool(rec["trigger_computed"])),
                         rec.get("merged_consistency"), rec.get("model_id") or rec.get("model"),
                         rec.get("prompt_version"), rec.get("time")))
    print(f"  {source}: {n_rec} records, {len(rows)} described circuits, {n_failed} failed (skipped), "
          f"{len(unmatched)} keys without a circuit, {len(bad_facet)} with a facet outside the allowed sets", flush=True)
    if bad_facet:
        raise RuntimeError(f"facet values outside TRIGGERS / CONDITIONS: {bad_facet[:5]}")

    path = descriptions_path(bundle)
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    conn = sqlite3.connect(tmp)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO circuit_description VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.execute("INSERT INTO description_fts(description_fts) VALUES ('rebuild')")
    st = os.stat(source)
    meta = dict(source=SOURCE.replace(os.sep, "/"), source_mtime=time.strftime("%Y-%m-%dT%H:%M:%S",
                                                                                 time.localtime(st.st_mtime)),
                source_bytes=st.st_size, n_records=n_rec, n_failed=n_failed, n_unmatched=len(unmatched),
                unmatched_keys=unmatched[:50], triggers=TRIGGERS, conditions=CONDITIONS,
                built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                note="written by Claude Sonnet (065 describe.py) from each circuit's reading data; hypotheses, "
                     "not verified")
    conn.executemany("INSERT INTO meta VALUES (?, ?)", [(k, _js(v)) for k, v in meta.items()])
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    _replace(tmp, path)

    from export.explorer.stage_finalize import Checks
    ck = Checks()
    validate(bundle, main, ck)
    cov = coverage(bundle, main)
    main.close()
    if ck.failures:
        raise RuntimeError(f"descriptions validation failed ({len(ck.failures)}): " + "; ".join(ck.failures))
    info = dict(source=meta["source"], records=n_rec, described=len(rows), failed=n_failed,
                unmatched=len(unmatched), coverage={k: cov[k] for k in ("circuits", "circuits_total")},
                triggers=dict(Counter(r[6] for r in rows)), conditions=dict(Counter(r[7] for r in rows)),
                secs=round(time.time() - t0, 1))
    _update_manifest(bundle, cov, info)
    return info
