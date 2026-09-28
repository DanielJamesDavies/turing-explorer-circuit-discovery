"""Reader for a 062 protocol run folder (the one holding ctx/ and main/)."""
from __future__ import annotations

import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional

import torch


def read_jsonl(pattern: str) -> List[dict]:
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


class RunSource:
    def __init__(self, run_dir: str, limit_targets: int = 0):
        self.dir = os.path.abspath(run_dir)
        main = os.path.join(self.dir, "main")
        self.status = {r["seed"]: r for r in read_jsonl(os.path.join(main, "status.shard*.jsonl"))}
        self.contexts = {r["seed"]: r for r in read_jsonl(os.path.join(main, "contexts.shard*.jsonl"))}
        self.train = {r["seed"]: r for r in read_jsonl(os.path.join(main, "train.shard*.jsonl"))}
        self.eval: Dict[str, Dict[str, dict]] = {}
        for r in read_jsonl(os.path.join(main, "eval.shard*.jsonl")):
            if not r.get("error"):
                self.eval.setdefault(r["seed"], {})[r["held"]] = r  # later rows win, as in pass_rule.py
        self.spec: Dict[str, Dict[str, dict]] = {}
        for r in read_jsonl(os.path.join(main, "spec.shard*.jsonl")):
            self.spec.setdefault(r["seed"], {})[r["pi"]] = r
        self.headline: Dict[str, dict] = {}
        tpath = os.path.join(self.dir, "targets.csv")
        if os.path.exists(tpath):
            with open(tpath, newline="", encoding="utf-8") as fh:
                self.headline = {r["seed"]: r for r in csv.DictReader(fh)}
        self.circuit_files = {os.path.basename(f)[:-3]: f for f in glob.glob(os.path.join(main, "circuits", "*.pt"))}
        self.ctx_files = {os.path.basename(f)[:-3]: f for f in glob.glob(os.path.join(self.dir, "ctx", "*.pt"))}
        keys = sorted(set(self.status) | set(self.ctx_files))
        if limit_targets:
            keys = keys[:limit_targets]
        self.targets = keys

    def ctx(self, key: str) -> Optional[dict]:
        f = self.ctx_files.get(key)
        return None if f is None else torch.load(f, map_location="cpu", weights_only=False)[key]

    def circuit(self, key: str) -> Any:
        f = self.circuit_files.get(key)
        if f is None:
            return None
        c = torch.load(f, map_location="cpu", weights_only=False)
        return next(iter(c.values())) if isinstance(c, dict) else c

    def target_status(self, key: str) -> tuple[str, Optional[str]]:
        """(status, reason): circuit | too_few_contexts | no_contrast | rejected."""
        st = self.status.get(key, {})
        if st.get("skip"):
            return st["skip"], st.get("skip")
        if key in self.circuit_files:
            return "circuit", None
        return "rejected", st.get("error") or "no circuit file"
