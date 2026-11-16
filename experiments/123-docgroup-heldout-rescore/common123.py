"""Shared pieces of 123 (DAN-217): paths, the corpus group labels, and the pure (CPU, testable) planning logic.

Group clause (the only change to the production held-out rule; the pilot 115's clause, unchanged):
  forbidden groups  the groups of every activating TRAINING row of the target (strong.train 48, which holds the fit's
                    D_train 32, + mid.train, which holds B_mid), each over all its corpus twins (same 085 content hash)
  a held-out row    is allowed only if none of its groups (over its twins) is forbidden
Group kinds:
  TuringLLM  doc    the source-file path with the phi-3-mini / data-augmentation prefix stripped = one Phi-3 original
                    plus its T5 paraphrase file (the paraphrase family; 114 check.groups())
             topic  the dirname of that path = the (up to 6) sibling documents of one leaf topic (stricter; sensitivity)
  public     doc    the record's own per-row `doc` (069 provenance/shard_<N>.docs.npy, the source document of the
                    window; -1 = unknown, then the row is its own group and never forbidden)

Every path is an env override; the defaults are this machine's unpacked pod archives (109 out/archives). On a pod set
TL_ROOT (the TuringLLM checkout, e.g. /root/turing-tl) and PUB_ROOT (the public checkout, e.g. /root/turing): the
defaults then follow the 082 chain's layout there (see README / RESUME).
"""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiments"
ARCH = Path(os.environ.get("ARCHIVES", str(EXP / "109-pod-run-analysis" / "out" / "archives")))
WORK = Path(os.environ.get("WORK123", str(HERE / "out")))

MODELS = ("turingllm", "pythia-70m", "gpt2-small")
GROUPS_OF = {"turingllm": ("doc", "topic"), "pythia-70m": ("doc",), "gpt2-small": ("doc",)}
VARIANTS_OF = {m: ("orig",) + g for m, g in GROUPS_OF.items()}      # orig = the production records, re-scored
FLOOR = 8                                                           # heldout_rule.FLOOR (reported, never dropped)
RULE123 = "123-docgroup/2026-11-16 (086 DAN-157 rule + group clause vs activating training rows incl. twins)"


def _tl_root():
    r = os.environ.get("TL_ROOT")
    return Path(r) if r else None


def _pub_root():
    r = os.environ.get("PUB_ROOT")
    return Path(r) if r else None


def tl_paths():
    """TuringLLM inputs (read only): fresh 094 records, the T1 scan, the T3 OUT (markers + main circuits), 085 corpus
    index, text_paths.txt."""
    r = _tl_root()
    if r is not None:
        base = dict(ctx=r / "experiments/082-public-runs/out_turingllm/ctx_fresh",
                    scan=r / "experiments/082-public-runs/out_turingllm/scan/scan_top512.pt",
                    out=r / "experiments/094-tl-v2-records/out_v2",
                    corpus=r / "experiments/085-heldout-audit/out",
                    text_paths=r / "data/text_paths.txt")
    else:
        base = dict(ctx=ARCH / "pub-turingllm-T2-records/root/turing-tl/experiments/082-public-runs/out_turingllm/ctx_fresh",
                    scan=ARCH / ("pub-turingllm-T1-scan/root/turing-tl/experiments/082-public-runs/out_turingllm/scan/"
                                 "scan_top512.pt"),
                    out=ARCH / "pub-turingllm-T3-main/root/turing-tl/experiments/094-tl-v2-records/out_v2",
                    corpus=EXP / "085-heldout-audit/out",
                    text_paths=REPO / "data/text_paths.txt")
    env = dict(ctx="TL_CTX", scan="TL_SCAN", out="TL_OUT_V2", corpus="TL_CORPUS", text_paths="TL_TEXT_PATHS")
    return {k: Path(os.environ[env[k]]) if os.environ.get(env[k]) else v for k, v in base.items()}


def pub_paths(model):
    """A public model's 082 OUT (ctx/ records, main/circuits, WCM markers) and its data / stores roots."""
    r = _pub_root()
    if r is not None:
        out = r / ("experiments/082-public-runs/out_%s" % model)
        data, stores = r / "data_public" / model, r / "outputs_public" / model
    else:
        out = ARCH / ("pub-%s-S10-results/experiments/082-public-runs/out_%s" % (model, model))
        data, stores = REPO / "data_public" / model, REPO / "outputs_public" / model
    key = model.upper().replace("-", "_")
    out = Path(os.environ.get("PUB_OUT_" + key, str(out)))
    return dict(out=out, ctx=Path(os.environ.get("PUB_CTX_" + key, str(out / "ctx"))), data=data, stores=stores)


def source_out(model):
    """The production driver OUT holding WCM_VERSION / WCM_R and main/circuits/<key>.pt."""
    return tl_paths()["out"] if model == "turingllm" else pub_paths(model)["out"]


def source_ctx(model):
    return tl_paths()["ctx"] if model == "turingllm" else pub_paths(model)["ctx"]


def variant_ctx(model, variant):
    """The CTX_DIR the re-score reads for one variant (orig = the production records themselves)."""
    return source_ctx(model) if variant == "orig" else WORK / model / ("ctx_%s" % variant)


def variant_out(model, variant):
    return WORK / model / ("rs_%s" % variant)


# ------------------------------------------------------------------------------------------------ TuringLLM groups
def tl_group_labels(text_paths):
    """(doc label, topic label) per text_paths.txt line, exactly as 114 check.groups() builds them (sorted relative paths
    -> doc ids; first-seen dirnames -> topic ids)."""
    paths = [p for p in Path(text_paths).read_text().split("\n") if p]
    rel = [p.split("/dataset/", 1)[1] for p in paths]
    doc_key = {r: k for k, r in enumerate(sorted(set(rel)))}
    top_key = {}
    doc = np.array([doc_key[r] for r in rel])
    topic = np.array([top_key.setdefault(os.path.dirname(r), len(top_key)) for r in rel])
    return doc, topic


class TLGroups:
    """Group labels per corpus id over all of the id's content twins (the 115 Groups interface: of(id, group))."""

    def __init__(self, corpus_dir=None, text_paths=None, src=None, hashes=None, labels=None):
        if src is None:
            p = tl_paths()
            corpus_dir = Path(corpus_dir or p["corpus"])
            src = np.load(corpus_dir / "corpus_src.npy")
            hashes = np.load(corpus_dir / "corpus_hash.npy")
            doc, topic = tl_group_labels(text_paths or p["text_paths"])
            labels = {"doc": doc, "topic": topic}
        self.src, self.hash, self.lab = np.asarray(src), np.asarray(hashes), labels
        hv = self.hash[1:]                                   # index 0 = no id (085 corpus_index)
        self.order = np.argsort(hv, kind="stable")
        self.sh = hv[self.order]
        self._cache = {}

    def twins(self, i):
        h = self.hash[int(i)]
        lo, hi = np.searchsorted(self.sh, h), np.searchsorted(self.sh, h, side="right")
        return self.order[lo:hi] + 1

    def of(self, i, g):
        k = (int(i), g)
        if k not in self._cache:
            self._cache[k] = set(int(x) for x in self.lab[g][self.src[self.twins(i)]])
        return self._cache[k]

    def clear(self):
        self._cache.clear()


# ------------------------------------------------------------------------------------------------ planning (pure)
def training_ids(rec):
    """Corpus ids of every activating training row: strong.train (incl. D_train) + mid.train (incl. B_mid)."""
    s, m = rec["strong"], rec.get("mid")
    ids = [int(s["ids"][j]) for j in s["train"]]
    if m is not None:
        ids += [int(m["ids"][j]) for j in m["train"]]
    return ids


def forbidden_groups(ids, group_of):
    out = set()
    for i in ids:
        out |= group_of(i)
    return out


def tl_plan(rec, scan_ids, group_of):
    """The CPU half of one TuringLLM target's rebuild (the group clause; no firing, no duplicate rule): the forbidden
    groups, which original held-out rows survive, and the allowed candidates in 086's order (original held rows in
    pool order, then the scan rows strongest first, rows of the record's own pool skipped). Mirrors 115 build_one's
    candidate loop exactly, so the counts here equal 115's counts for the same record."""
    s = rec["strong"]
    forb = forbidden_groups(training_ids(rec), group_of)
    pool_ids = set(int(x) for x in s["ids"])
    held_sorted = sorted(int(x) for x in s["held"])
    orig_ok = [j for j in held_sorted if not (group_of(int(s["ids"][j])) & forb)]
    scan_total, scan_ok = 0, []
    for r, i in enumerate(int(x) for x in scan_ids):
        if i in pool_ids:
            continue
        scan_total += 1
        if not (group_of(i) & forb):
            scan_ok.append((r + 1, i))
    return dict(forbidden=forb, orig_kept_idx=orig_ok, scan_allowed=scan_ok,
                counts=dict(orig_total=len(held_sorted), orig_allowed=len(orig_ok), scan_total=scan_total,
                            scan_allowed=len(scan_ok)))


def leak_counts(rec, group_of, forb=None):
    """Held-out rows sharing a group with the target's activating training rows (strong held, mid held), and contrast
    held-out rows (rec["neg_ids"][48:]) sharing a group with the contrast-train rows or with the activating training
    rows. Reported, not a rule, except strong (the rebuild) and mid (MID_RULE=drop)."""
    s, m = rec["strong"], rec.get("mid")
    if forb is None:
        forb = forbidden_groups(training_ids(rec), group_of)
    out = dict(strong_held=len(s["held"]),
               strong_leaky=sum(bool(group_of(int(s["ids"][j])) & forb) for j in s["held"]))
    if m is not None:
        out.update(mid_held=len(m["held"]), mid_leaky=sum(bool(group_of(int(m["ids"][j])) & forb) for j in m["held"]))
    neg_ids = rec.get("neg_ids")
    if neg_ids is not None and len(neg_ids):
        n = len(neg_ids)
        n_tr = max(1, n - int(round(n * 0.25)))               # the scorers' contrast split (086 neg_split)
        held_c = [int(x) for x in neg_ids[n_tr:] if int(x) > 0]
        cforb = forbidden_groups([int(x) for x in neg_ids[:n_tr] if int(x) > 0], group_of)
        out.update(contrast_held=len(held_c),
                   contrast_leaky_vs_contrast_train=sum(bool(group_of(i) & cforb) for i in held_c),
                   contrast_leaky_vs_activating_train=sum(bool(group_of(i) & forb) for i in held_c))
    return out


def drop_mid(rec, group_of, forb):
    """MID_RULE=drop: remove the mid-band held-out rows in a forbidden group (no refill: the mid pool has no top-up
    source). If every mid held row is leaky, the set is kept as it is and flagged (the driver cannot score an empty
    set; the report drops that target's mid row). Returns (new mid dict or None = unchanged, info)."""
    m = rec.get("mid")
    if m is None:
        return None, dict(mid_rule="none")
    keep = [int(j) for j in m["held"] if not (group_of(int(m["ids"][j])) & forb)]
    info = dict(mid_rule="drop", mid_held_old=len(m["held"]), mid_held_new=len(keep), mid_below_floor=len(keep) < FLOOR)
    if not keep:
        info.update(mid_held_new=len(m["held"]), mid_all_leaky_kept=True)
        return None, info
    if len(keep) == len(m["held"]):
        return None, info
    m2 = dict(m)
    m2["held"] = keep
    return m2, info


# ------------------------------------------------------------------------------------------------ public (removal)
def pub_doc_of(rec, pool):
    return [int(x) for x in rec[pool].get("doc", [])]


def pub_rebuild(rec, mid_rule="drop"):
    """A public record's document-disjoint held-out sets by REMOVAL (the public path has no top-up source, as its
    group-aware split): strong.held and (mid_rule drop) mid.held lose every row whose document supplies an activating
    training row (strong.train + mid.train). Unknown documents (-1) are never forbidden. Training-side fields are
    untouched. Returns (new record, summary)."""
    s, m = rec["strong"], rec.get("mid")
    sd = pub_doc_of(rec, "strong")
    md = pub_doc_of(rec, "mid") if m is not None else []
    forb = {sd[j] for j in s["train"]} | ({md[j] for j in m["train"]} if m is not None else set())
    forb.discard(-1)
    keep = [int(j) for j in s["held"] if sd[j] not in forb]
    r2 = dict(rec)
    s2 = dict(s)
    s2["held"] = keep
    r2["strong"] = s2
    info = dict(n_want=len(s["held"]), n_held=len(keep), n_old_leaky=len(s["held"]) - len(keep),
                n_orig_kept=len(keep), n_scan=0, below_floor=len(keep) < FLOOR, unscorable=len(keep) == 0,
                n_forbidden_groups=len(forb), unknown_doc_held=sum(sd[j] < 0 for j in s["held"]))
    if m is not None:
        mkeep = [int(j) for j in m["held"] if md[j] not in forb]
        info.update(mid_held_old=len(m["held"]), mid_leaky=len(m["held"]) - len(mkeep), mid_rule=mid_rule)
        if mid_rule == "drop" and mkeep and len(mkeep) < len(m["held"]):
            m2 = dict(m)
            m2["held"] = mkeep
            r2["mid"] = m2
        info["mid_held_new"] = len(r2["mid"]["held"])
        info["mid_all_leaky_kept"] = mid_rule == "drop" and not mkeep and len(m["held"]) > 0
    held_ids = [int(s["ids"][j]) for j in keep]
    r2["docdisjoint"] = dict(version=RULE123, group="doc", rule="removal", held_ids=held_ids,
                             old_held_ids=[int(s["ids"][j]) for j in s["held"]], **info)
    return r2, info


def check_disjoint(rec, group_of):
    """The split check on a built record: (# strong held rows sharing a group with an activating training row,
    # duplicate held rows). Both must be 0."""
    forb = forbidden_groups(training_ids(rec), group_of)
    s = rec["strong"]
    held = [int(s["ids"][j]) for j in s["held"]]
    return sum(bool(group_of(i) & forb) for i in held), len(held) - len(set(held))


# ------------------------------------------------------------------------------------------------ io helpers
def read_targets(path):
    return [t for t in Path(path).read_text().split() if t]


def done_keys(path, field="seed"):
    """Keys of the complete rows of a jsonl (a truncated last line after a kill is skipped; error rows are not done)."""
    out = set()
    p = Path(path)
    for f in [p] + sorted(p.parent.glob(p.stem + ".shard*.jsonl")):
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r:
                out.add(r.get(field))
    return out


def read_jsonl(path):
    p = Path(path)
    rows = []
    for f in [p] + sorted(p.parent.glob(p.stem + ".shard*.jsonl")):
        if f.exists():
            for line in open(f, encoding="utf-8"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def open_append(path):
    """Append handle on a jsonl that first terminates a torn last line (a kill mid-write), so the next row is not
    glued onto it and lost."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size:
        with open(p, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            torn = fh.read(1) != b"\n"
        if torn:
            with open(p, "a", encoding="utf-8") as fh:
                fh.write("\n")
    return open(p, "a", encoding="utf-8")


def shard_path(path, shard_i, shard_k):
    p = Path(path)
    return p if shard_k == 1 else p.with_name("%s.shard%d.jsonl" % (p.stem, shard_i))


def save_atomic(obj, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp-%d" % os.getpid())
    torch.save(obj, tmp)
    os.replace(tmp, path)


def band_tl(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")
