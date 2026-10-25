"""Public-model smoke for the group-aware split (DAN-157): build_record with and without group_split on the local 069
smoke stores (outputs_public/<family>, data_public/<family>), the targets of 069's records_check.json.

Checks per record: no group straddles the split; the group record's training side equals what the split says; every
held-out row passes the rule against the training side, recomputed with 086 rebuild.py's independent predicates
(exact / anchor-prefix / near; duplicates only, no document rule since 2026-10-24); the strongest held-out set reaches
16 (reported per record); the default record is the pre-DAN-157 one (stratified per-row split). Document sharing is
reported (provenance), not a failure.
Writes out/public_smoke/<family>.json. CPU or GPU.

  FAMILY=pythia-70m PYTHONPATH=src python experiments/086-heldout-rebuild/public_smoke.py
  env: FAMILY (pythia-70m | gpt2-small)  DATA / STORES (default data_public/<FAMILY> / outputs_public/<FAMILY>)
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
FAMILY = os.environ.get("FAMILY", "pythia-70m")
DATA = os.environ.get("DATA", str(REPO / "data_public" / FAMILY))
STORES = os.environ.get("STORES", str(REPO / "outputs_public" / FAMILY))


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    C = _mod("data_pass_069_common", REPO / "experiments" / "069-public-data-pass" / "common.py")
    cfg = C.configure(FAMILY, DATA, STORES)
    import torch
    from circuit.context_split import stratified_split
    from circuit.protocol_contexts import build_record, group_split_for, make_selector
    from circuit.protocol_contexts import skip_reason as PC_skip
    from data.loader import DataLoader
    from model.inference import Inference
    from sae.bank import SAEBank
    from store.latent_stats import latent_stats
    RB = _mod("rebuild_086", HERE / "rebuild.py")
    device = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu")
    loader = DataLoader(device=device)
    C.load_stores(os.path.abspath(STORES), int(sum(loader._shard_sequence_counts)))
    latent_stats.load(os.path.join(STORES, "latent_stats.pt"))
    inference, bank = Inference(device=device), SAEBank(device=device)
    gs = group_split_for(DATA, loader)
    docs = gs.docs if gs.docs is not None else (lambda ids: np.full(len(list(ids)), -1))
    targets = [r["target"] for r in json.load(open(os.path.join(STORES, "records_check.json")))["records"]]
    kinds = list(bank.kinds)
    v1 = cfg.discovery.context_v1
    rows = []
    for t in targets:
        l, kd, i = t.split(".")
        comp = int(l) * len(kinds) + kinds.index(kd)
        t0 = time.time()
        base = build_record(inference, bank, loader, comp, int(i), selector=make_selector(inference, bank, loader), cfg=v1)
        grp = build_record(inference, bank, loader, comp, int(i), selector=make_selector(inference, bank, loader), cfg=v1,
                           group_split=gs)
        row = dict(target=t, skip=base.get("strong") is None, seconds=round(time.time() - t0, 1))
        if base.get("strong") is None:
            rows.append(row); print(row, flush=True); continue
        if grp.get("strong") is None:
            row.update(ok=False, failures=["group record skipped: %s" % PC_skip(grp)], held={})
            rows.append(row); print(row, flush=True); continue
        fail, notes = [], []
        for name in ("strong", "mid"):
            b, g = base.get(name), grp.get(name)
            if b is None:
                continue
            if (b["train"], b["held"]) != stratified_split(b["peak"]):
                fail.append("%s: default split is not the stratified split" % name)
            gr = g["group"]
            if {gr[j] for j in g["train"]} & {gr[j] for j in g["held"] + g["unused"]}:
                fail.append("%s: a group straddles the split" % name)
            if sorted(g["train"] + g["held"] + g["unused"]) != list(range(len(g["ids"]))):
                fail.append("%s: train / held / unused do not partition the pool" % name)
        s, m = grp["strong"], grp.get("mid")
        if m is not None:                              # joint groups: no straddle ACROSS the pools either
            lab = lambda d, ix: {d["group"][j] for j in ix}
            if lab(s, s["train"]) & lab(m, m["held"] + m["unused"]) or lab(m, m["train"]) & lab(s, s["held"]):
                fail.append("a group straddles the split across the pools")
        ndoc = {int(x) for x in docs(list(grp.get("neg_ids") or []))} - {-1}
        hdoc = {int(d["doc"][j]) for d in (s, m) if d is not None for j in d["held"]} - {-1}
        if ndoc & hdoc:                                # provenance only (no document rule)
            notes.append("contrast rows share %d documents with held-out rows" % len(ndoc & hdoc))
        F = [s["pos"][s["train"]].numpy()] + ([m["pos"][m["train"]].numpy()] if m is not None else []) + [grp["neg"].numpy()]
        F = np.concatenate(F).astype(np.int64)
        fids = [s["ids"][j] for j in s["train"]] + ([m["ids"][j] for j in m["train"]] if m is not None else [])
        fids += list(grp.get("neg_ids") or [])
        fdoc = {int(d) for d in docs(fids) if int(d) >= 0}
        held_stats = {}
        for name, d in (("strong", s), ("mid", m)):
            if d is None:
                continue
            acc = []
            for j in sorted(int(x) for x in d["held"]):
                h, a = d["pos"][j].numpy().astype(np.int64), int(d["arg"][j])
                v = RB.violations(h, a, F) + (["vs held " + x for x in RB.violations(h, a, np.asarray(acc))] if acc else [])
                if int(d["doc"][j]) in fdoc:
                    notes.append("%s %d: document of a training-side row" % (name, j))
                if not float(d["peak"][j]) > 0:
                    v.append("silent")
                if v:
                    fail.append("%s %d: %s" % (name, j, v))
                acc.append(h)
            hv = grp["heldout"][name]
            # which training-side role shares a document with the split's held-out rows (before the rule)
            role_docs = dict(strong_train={int(x) for x in docs([s["ids"][j] for j in s["train"]])},
                             mid_train={int(x) for x in docs([m["ids"][j] for j in m["train"]])} if m is not None else set(),
                             contrast={int(x) for x in docs(list(grp.get("neg_ids") or []))})
            split_held = [j for j in range(len(d["ids"])) if j not in set(d["train"])]
            hd = [int(x) for x in docs([d["ids"][j] for j in split_held])]
            by_role = {r: sum(1 for x in hd if x >= 0 and x in v) for r, v in role_docs.items()}
            only_ctr = sum(1 for x in hd if x in role_docs["contrast"] and x not in role_docs["strong_train"]
                           and x not in role_docs["mid_train"])
            held_stats[name] = dict(n_default=len(base[name]["held"]), n_split=hv["n_held_split"], n_held=hv["n_held"],
                                    n_train=len(d["train"]), n_unused=len(d["unused"]),
                                    n_contrast=int(grp["neg"].shape[0]), n_contrast_default=int(base["neg"].shape[0]),
                                    contrast_excluded_ids=grp.get("contrast_excluded_ids"),
                                    below_floor=hv["below_floor"], n_groups=hv["n_groups"], max_group=hv["max_group"],
                                    dropped=hv["reasons"], doc_shared_with=by_role, doc_only_contrast=only_ctr)
        row.update(held=held_stats, failures=fail, ok=not fail, n_doc_notes=len(notes),
                   strong_held_16=held_stats.get("strong", {}).get("n_held") == 16,
                   training_side_changed=bool(s["train"] != base["strong"]["train"]))
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = HERE / "out" / "public_smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("%s.json" % FAMILY)).write_text(json.dumps(dict(family=FAMILY, records=rows), indent=1))
    ok = [r for r in rows if not r["skip"]]
    print("%s: %d records, %d usable, all checks pass: %s; strongest held-out = 16 on %d / %d; strongest held-out "
          "sizes %s" % (FAMILY, len(rows), len(ok), all(r["ok"] for r in ok), sum(1 for r in ok if r.get("strong_held_16")),
                        len(ok), [r.get("held", {}).get("strong", {}).get("n_held") for r in ok]))


if __name__ == "__main__":
    main()
