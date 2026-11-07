"""Experiment 081 (task circuits): the pure logic, with no model and no GPU. Shared by make_tasks.py, prepare.py,
probe.py, calibrate.py, run.py, report.py and figure.py, and unit-tested by the test_081_*.py files. (Named
task_common, not common, so that one pytest session over 080 and 081 does not import one experiment's common.py for
the other's.)

  model          MODEL=gpt2-small (default, 2026-10-21) or turingllm; tasks live in tasks/<model>/
  task registry  every task's source, family, metrics, contrast source (the dataset's own patch prompts, or a
                 generated "close, silent" contrast set), citation and whether the model is expected to have it
  prompts        right-padding, per-row anchors, valid-position masks, answers as token SEQUENCES (multi-token answers
                 teacher-forced as (good, bad) row pairs) or token SETS (greater-than)
  splits         full datasets, natural split (2026-10-21): the 75 / 25 split by group (fixed datasets without an
                 official split), disjoint list halves (generated tasks), the held-out cap (HELDOUT_CAP), the run-time
                 held-out subset (HELDOUT_N, smoke), the training stream (presentation_order: without replacement
                 within an epoch)
  competence     the pre-stated competence rule (probe.py, held-out prompts)
  contrast sets  the close-silent rule for tasks without patch prompts: rank candidates by cosine of mean-pooled
                 final-layer residuals, keep those on which the item's good answer is NOT preferred, pool 64
  run config     the lambda grids and lr the main run reads (config_main.json, written by calibrate.py ADOPT=1)
  calibration    failure rate, monotonicity, the grid recommendation
  curves         per task and arm: (size, faithfulness, pass) over lambda; claim-2 ratios (080's helpers reused)
  random null    support draws matched to W's per-site counts
  structure      role- and head-resolved member profiles -> the registered T4 checks against the published GPT-2
                 circuits (IOI: Wang et al.; greater-than: Hanna et al.) and the agreement checks (secondary)
  link to 062    T5 (descriptive; TuringLLM only: the 062 production targets are TuringLLM latents)
  verdicts       T1-T5 as registered in the README (2026-10-21)
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
EXTERNAL = ROOT / "data" / "external"
MODELS = ("gpt2-small", "turingllm")
MODEL = os.environ.get("MODEL", "gpt2-small")
if MODEL not in MODELS:
    raise ValueError("MODEL must be one of %s, got %r" % (MODELS, MODEL))


def task_dir(model: str = MODEL) -> Path:
    return HERE / "tasks" / model


TASK_DIR = task_dir()


def _load_by_path(name: str, path: Path):
    """A sibling experiment's module, loaded by path under its own name (putting its folder on sys.path would let
    its common.py / report.py shadow this experiment's)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C80 = _load_by_path("_common_080", EXP / "080-latent-vs-output" / "common.py")

# ------------------------------------------------------------------------------------------- protocol constants
GAMMA = C80.GAMMA                 # gamma_C = gamma_A = 0.25 (protocol v1)
PROD_LR, PROD_WD, PROD_STEPS = C80.PROD_LR, C80.PROD_WD, C80.PROD_STEPS
BAND_LO, BAND_HI, NEC_MIN = C80.BAND_LO, C80.BAND_HI, C80.NEC_MIN
LOGIT_NORM = "gap"                # as 080: each data term ~ (1 - faithfulness)^2
# (2026-10-27) "gap_term" = each ablation term by its own closed-mask gap^2 (learned_mask logit_norm; the A/B of the
# v2 local run). Any value of the run config's / calibrate.py's logit_norm is passed through to the engine.
LOGIT_NORMS = ("gap", "gap_term")
# WCM protocol version (2026-10-27; paper/protocol.md section 12). v2 = the A and C training terms under the
# scorer's sparsity-preserving fill with straight-through gradients (learned_mask train_fill / sparse_grad), as 062's
# driver.WCM_VERSIONS. Protocol v2's third field, train_means_source="eval_ref" (train against the scorer's own A / C
# means), holds BY CONSTRUCTION here: engine.means(B) computes the A / C means once per task and the SAME tensors are
# passed to the fit (floor_means / pos_floor_means) and to the scorer (means_A / means_C), so there is nothing to
# switch (the src field is read only by ablation_gradient's discovery wrapper, which 081 does not use).
WCM_VERSIONS = {
    "v1": dict(train_fill="dense", sparse_grad="ste"),
    "v2": dict(train_fill="sparse", sparse_grad="ste"),
}
WCM_VERSION = os.environ.get("WCM_VERSION", "v2")
if WCM_VERSION not in WCM_VERSIONS:
    raise SystemExit("WCM_VERSION must be one of %s, got %r" % (sorted(WCM_VERSIONS), WCM_VERSION))
weight_decay = C80.weight_decay   # lr x wd held at production's 0.05 x 0.05
lam_tag = C80.lam_tag
num = C80.num
interp_log_n = C80.interp_log_n
size_to_reach = C80.size_to_reach
overlap = C80.overlap
monotone = C80.monotone

TASK_SEED = 20261019
HOLDOUT_FRAC = 0.25               # 75 / 25 for fixed datasets without an official split (by group)
PAD_ID = 0                        # right-padding only (left-padding with 0 corrupts TuringLLM; 047's padding trap)
# Full datasets, natural split (Daniel, 2026-10-21). Training = the whole training split; generated tasks (IOI,
# greater-than, induction) get N_TRAIN_GEN training and N_HELD_GEN held-out items from DISJOINT halves of their
# name / object / noun / token lists (LIST_HOLD_FRAC); every held-out set is capped at HELDOUT_CAP by fixed-seed
# sampling (0 = no cap; the uncapped size is recorded in the task file).
N_TRAIN_GEN = 1600
N_HELD_GEN = 5000
LIST_HOLD_FRAC = 0.5              # the held-out side's share of a generated task's lists (held-out > training here)
HELDOUT_CAP = int(os.environ.get("HELDOUT_CAP", 5000))
PROD_BATCH = 4                    # protocol v1's batch; with PROD_STEPS (400): 1,600 training presentations per fit
N_DRAWS = 5                       # random-null draws per task per W lambda
FAITH = ("free0", "freeM_topk", "freeN_topk")                 # Z, A_sp, C_sp
PRIMARY = "task"                  # the metric the pass rule is applied to (the paper's task metric)
OBJ = "obj"                       # the metric the fit reproduces
ARMS: Dict[str, Dict[str, object]] = {
    "W": dict(weighted=True, random=False, c_term=True),       # WCM: fitted coefficients, Z + C + A
    "U": dict(weighted=False, random=False, c_term=True),      # unweighted: alpha = 1 (claim 2)
    "RND": dict(weighted=True, random=True, c_term=True),      # random support size-matched to W, no penalty
    # sensitivity arm (generated-contrast tasks only): W trained on Z + A, NO C term (the generated contrast set is
    # Daniel's latent rule transferred to outputs; this arm shows what the fit does without it)
    "W_ZA": dict(weighted=True, random=False, c_term=False),
}
FIT_ARMS = ("W", "U")             # arms with their own lambda grid in the run config (RND and W_ZA use W's)

# ------------------------------------------------------------------------------------------- task registry
CITE = {
    "wang2022": "Wang, Variengien, Conmy, Shlegeris & Steinhardt (2022), Interpretability in the wild: a circuit for "
                "indirect object identification in GPT-2 small, ICLR 2023",
    "hanna2023": "Hanna, Liu & Variengien (2023), How does GPT-2 compute greater-than?, NeurIPS",
    "marks2025": "Marks, Rager, Michaud, Belinkov, Bau & Mueller (2025), Sparse feature circuits, ICLR",
    "mathwin2023": "Mathwin, Corlouer, Kran, Barez & Nanda (2023), Identifying a preliminary circuit for predicting "
                   "gendered pronouns in GPT-2 small (Apart hackathon report)",
    "meng2022": "Meng, Bau, Andonian & Belinkov (2022), Locating and editing factual associations in GPT (ROME), "
                "NeurIPS",
    "olsson2022": "Olsson et al. (2022), In-context learning and induction heads, Transformer Circuits Thread",
    "warstadt2020": "Warstadt, Parrish, Liu, Mohananey, Peng, Wang & Bowman (2020), BLiMP: the benchmark of linguistic "
                    "minimal pairs for English, TACL",
}
# contrast: "original" = the dataset's own patch / counterfactual prompts are the C ablation's mean;
#           "generated" = a close, silent contrast pool from the same dataset (prepare.py; task_common.build_pool)
_TOKEN = dict(task="logit_diff", obj="logit_diff", logit_metric="token", vac=dict(task=0.1, obj=0.1))
TASKS: Dict[str, Dict] = {
    "ioi": dict(family="ioi", cite="wang2022", expect=True, contrast="original", source="ioi_easy_transformer",
                desc="Wang et al.'s BABA / ABBA templates -> IO vs S; C = their ABC distribution", **_TOKEN),
    "greater_than": dict(family="greater_than", cite="hanna2023", expect=True, contrast="original",
                         source="greater_than_hanna", task="set_probdiff", obj="set_logmass", logit_metric="set",
                         vac=dict(task=0.05, obj=0.1),
                         desc="The {noun} lasted from the year {XXYY} to the year {XX} -> YY' > YY vs <= YY; "
                              "C = XX01"),
    "sva_simple": dict(family="sva", cite="marks2025", expect=True, contrast="original",
                       source="sfc_feature_circuits", desc="SFC simple: The {subj} -> verb", **_TOKEN),
    "sva_within_rc": dict(family="sva", cite="marks2025", expect=True, contrast="original",
                          source="sfc_feature_circuits", desc="SFC within_rc: The {s1} that the {s2} -> RC verb",
                          **_TOKEN),
    "sva_across_rc": dict(family="sva", cite="marks2025", expect=True, contrast="original",
                          source="sfc_feature_circuits", desc="SFC rc: The {s1} that the {s2} {v} -> main verb",
                          **_TOKEN),
    "sva_across_pp": dict(family="sva", cite="marks2025", expect=True, contrast="original",
                          source="sfc_feature_circuits", desc="SFC nounpp: The {s} {prep} the {n} -> verb", **_TOKEN),
    "gender_pronoun": dict(family="gender", cite="mathwin2023", expect=True, contrast="original",
                           source="mathwin_gendered_pronoun",
                           desc="So {name} is a really great friend, isn't -> he vs she; C = 'That person ...'",
                           **_TOKEN),
    "factual_known": dict(family="factual", cite="meng2022", expect=None, contrast="generated",
                          source="rome_known_1000", desc="ROME known_1000: prompt -> true object vs another "
                                                         "object of the same relation", **_TOKEN),
    "induction": dict(family="induction", cite="olsson2022", expect=True, contrast="original", source=None,
                      desc="R + R[:j+1] (random tokens) -> R[j+1] vs R'[j+1]; C = R' + R[:j+1]", **_TOKEN),
}
BLIMP_PREFIX = "blimp_"


def task_spec(name: str) -> Dict:
    """The registry entry of a task; BLiMP subsets (blimp_<UID>) share one entry."""
    if name in TASKS:
        return dict(TASKS[name], name=name)
    if name.startswith(BLIMP_PREFIX):
        return dict(family="blimp", cite="warstadt2020", expect=None, name=name, contrast="generated",
                    source="blimp", desc="BLiMP %s: shared prefix -> critical word, good vs bad"
                    % name[len(BLIMP_PREFIX):], **_TOKEN)
    raise KeyError("unknown task %r" % name)


def answer_mode(task: Mapping) -> str:
    """"set" (token sets, greater-than), "seq" (some answer has more than one token: teacher-forced pairs) or
    "token" (every answer is one token)."""
    if task.get("logit_metric") == "set":
        return "set"
    if any(len(r["good"]) > 1 or len(r["bad"]) > 1 for r in task["rows"]):
        return "seq"
    return "token"


def metric_kinds(mode: str, spec: Mapping) -> Dict[str, str]:
    """The (task, obj) metric kinds and the engine's logit_metric for an answer mode."""
    if mode == "seq":
        return dict(task="seq_diff", obj="seq_diff", logit_metric="seq")
    return dict(task=spec["task"], obj=spec["obj"], logit_metric=spec["logit_metric"])


def task_files(d: Optional[Path] = None) -> List[Path]:
    d = Path(d) if d is not None else TASK_DIR
    return sorted(p for p in d.glob("*.json") if p.stem not in ("summary", "blimp_survey")
                  and not p.stem.endswith(".contrast"))


def load_task(name: str, d: Optional[Path] = None) -> Dict:
    p = (Path(d) if d is not None else TASK_DIR) / ("%s.json" % name)
    if not p.exists():
        raise FileNotFoundError("%s does not exist: run make_tasks.py (MODEL=%s) first" % (p, MODEL))
    return json.loads(p.read_text())


def contrast_path(name: str, d: Optional[Path] = None) -> Path:
    return (Path(d) if d is not None else TASK_DIR) / ("%s.contrast.json" % name)


def load_contrast(name: str, d: Optional[Path] = None) -> Dict:
    p = contrast_path(name, d)
    if not p.exists():
        raise FileNotFoundError("%s does not exist: task %s has no patch prompts of its own, so its C ablation needs "
                                "the generated contrast set; run prepare.py (GPU) first" % (p, name))
    return json.loads(p.read_text())


# ------------------------------------------------------------------------------------------- prompts and split
def split_groups(groups: Sequence[str], frac: float = HOLDOUT_FRAC, seed: int = TASK_SEED) -> Tuple[List[int], List[int]]:
    """75 / 25 split BY GROUP (rows sharing a group stay on one side), fixed seed: the sorted distinct groups are
    shuffled and the first round(n_groups x frac) are held out. Returns (train, held) row indices, each in a seeded
    shuffled order (the engine cycles its micro-batches through the training rows in order)."""
    uniq = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_hold = int(round(len(uniq) * frac))
    held_g = set(uniq[:n_hold])
    tr = [i for i, g in enumerate(groups) if g not in held_g]
    ho = [i for i, g in enumerate(groups) if g in held_g]
    rng.shuffle(tr)
    rng.shuffle(ho)
    return tr, ho


def split_list(items: Sequence, frac: float = LIST_HOLD_FRAC, seed=TASK_SEED, tag: str = "") -> Tuple[List, List]:
    """A source list (names, objects, nouns, vocabulary tokens) split into DISJOINT (train, held) halves, fixed seed:
    the sorted distinct items are shuffled and the first round(n x frac) go to the held-out side (MIB's private test
    set uses held-out names the same way). Both sides keep the source's order."""
    uniq = sorted(set(items), key=lambda x: (str(type(x)), x))
    rng = random.Random("%s-list-%s" % (tag, seed))
    rng.shuffle(uniq)
    n_hold = int(round(len(uniq) * frac))
    if not 0 < n_hold < len(uniq):
        raise ValueError("list %r of %d items is too short to split" % (tag, len(uniq)))
    held = set(uniq[:n_hold])
    return [x for x in items if x not in held], [x for x in items if x in held]


def cap_held(held: Sequence[int], cap: Optional[int] = None, seed: str = "") -> Tuple[List[int], int]:
    """The held-out set capped at `cap` items (HELDOUT_CAP by default; 0 or None = no cap) by fixed-seed sampling
    (random.Random(seed).sample: the kept items in a seeded order). Returns (kept, uncapped size)."""
    cap = HELDOUT_CAP if cap is None else int(cap)
    held = list(held)
    if not cap or len(held) <= cap:
        return held, len(held)
    return random.Random("%s-cap" % seed).sample(held, cap), len(held)


def heldout_n() -> int:
    """HELDOUT_N (env; 0 = all): use only the first N held-out items at run time (smoke.sh). The held-out order is
    seeded, so the first N are a fixed random subset."""
    return int(os.environ.get("HELDOUT_N", 0) or 0)


def presentation_order(n_items: int, n_present: int, seed: str = "") -> List[int]:
    """The training stream of one fit: n_present item indices (0..n_items-1) drawn WITHOUT replacement until the pool
    is exhausted, then reshuffled (one seeded permutation per epoch, concatenated, truncated). A pool of >= n_present
    items is never repeated. The same seed gives every fit of a task the same stream."""
    if n_items < 1 or n_present < 1:
        raise ValueError("presentation_order needs a non-empty pool and at least one presentation")
    rng = random.Random("%s-present" % seed)
    out: List[int] = []
    while len(out) < n_present:
        perm = list(range(n_items))
        rng.shuffle(perm)
        out += perm
    return out[:n_present]


def pad_right(seqs: Sequence[Sequence[int]], width: Optional[int] = None, pad_id: int = PAD_ID):
    """(tokens [N, T] right-padded, anchors [N] = each prompt's last real token, valid [N, T]) as lists."""
    width = max(len(s) for s in seqs) if width is None else int(width)
    if any(len(s) == 0 or len(s) > width for s in seqs):
        raise ValueError("every prompt needs 1..%d tokens" % width)
    toks = [list(s) + [pad_id] * (width - len(s)) for s in seqs]
    anchors = [len(s) - 1 for s in seqs]
    valid = [[True] * len(s) + [False] * (width - len(s)) for s in seqs]
    return toks, anchors, valid


def pad_sets(sets: Sequence[Sequence[int]]) -> List[List[int]]:
    """Token sets or sequences as one rectangular list, padded with -1."""
    k = max(len(s) for s in sets)
    if any(len(s) == 0 for s in sets):
        raise ValueError("an empty token set / answer")
    return [list(s) + [-1] * (k - len(s)) for s in sets]


def task_arrays(task: Mapping, rows: Sequence[int], mode: Optional[str] = None) -> Dict:
    """The rows' arrays (lists).

    token / set mode: one row per item; tokens = the prompt, anchors = its last token, good / bad = padded sets.
    seq mode: TWO rows per item, (good, bad) interleaved: row = prompt + answer[:-1] (teacher forcing), anchor = the
      prompt's last token, answers = the row's answer tokens (-1 padded); valid marks the PROMPT positions only (the
      A / C means are over prompts, not over teacher-forced answer tokens). n_rows = 2 x items.
    Counterfactual prompts (cf_tokens etc.) are included when every row has one (contrast = "original")."""
    mode = mode or answer_mode(task)
    R = [task["rows"][i] for i in rows]
    has_cf = all(r.get("cf_tokens") for r in R)
    if mode == "seq":
        seqs, prompt_len, answers = [], [], []
        for r in R:
            for ans in (r["good"], r["bad"]):
                seqs.append(list(r["tokens"]) + list(ans[:-1]))
                prompt_len.append(len(r["tokens"]))
                answers.append(list(ans))
        width = max([len(s) for s in seqs] + ([len(r["cf_tokens"]) for r in R] if has_cf else []))
        t, _, _ = pad_right(seqs, width)
        anchors = [n - 1 for n in prompt_len]
        valid = [[j < n for j in range(width)] for n in prompt_len]
        out = dict(mode=mode, tokens=t, anchors=anchors, valid=valid, answers=pad_sets(answers), width=width,
                   n_items=len(R), rows_per_item=2)
    else:
        width = max(max(len(r["tokens"]) for r in R), max((len(r["cf_tokens"]) for r in R), default=0)
                    if has_cf else 0)
        t, a, v = pad_right([r["tokens"] for r in R], width)
        out = dict(mode=mode, tokens=t, anchors=a, valid=v, good=pad_sets([r["good"] for r in R]),
                   bad=pad_sets([r["bad"] for r in R]), width=width, n_items=len(R), rows_per_item=1)
    if has_cf:
        ct, ca, cv = pad_right([r["cf_tokens"] for r in R], out["width"])
        out.update(cf_tokens=ct, cf_anchors=ca, cf_valid=cv)
    return out


# ------------------------------------------------------------------------------------------- competence
COMPETENCE = dict(min_mean=0.0, min_frac_correct=0.70, min_cf_gap=0.0)


def competence(clean: Sequence[float], cf: Optional[Sequence[float]], rule: Mapping = COMPETENCE) -> Dict:
    """The pre-stated competence rule on the held-out items (metric per item: clean prompt, and its matched
    counterfactual scored for the clean item's good / bad answers): COMPETENT iff
      (a) mean(clean) > 0,
      (b) the share of items with clean metric > 0 is >= 0.70, and
      (c) mean(clean) - mean(cf) > 0, only where the dataset has its own counterfactuals; cf=None (generated-
          contrast tasks, whose contrast prompts are silent by construction) marks (c) not applicable."""
    c = [float(x) for x in clean]
    if not c:
        return dict(n=0, competent=False, reason="no prompts")
    mean_c = sum(c) / len(c)
    frac = sum(1 for x in c if x > 0) / len(c)
    a, b = mean_c > rule["min_mean"], frac >= rule["min_frac_correct"]
    if cf is None:
        return dict(n=len(c), mean_clean=mean_c, mean_cf=None, frac_correct=frac, cf_gap=None, crit_mean=a,
                    crit_frac=b, crit_cf=None, competent=bool(a and b))
    f = [float(x) for x in cf]
    mean_f = sum(f) / len(f) if f else float("nan")
    gap = mean_c - mean_f if f else float("nan")
    cc = (gap == gap) and gap > rule["min_cf_gap"]
    return dict(n=len(c), mean_clean=mean_c, mean_cf=mean_f, frac_correct=frac, cf_gap=gap, crit_mean=a,
                crit_frac=b, crit_cf=cc, competent=bool(a and b and cc))


# ------------------------------------------------------------------------------------------- contrast sets
POOL_SIZE = 64                    # the C ablation's contrast pool per task (protocol v1 uses 64 contrast contexts)
SILENT_MAX = 0.0                  # "silent" for an output endpoint: metric(good_i, bad_i | candidate) <= 0
TOP_K_CANDIDATES = 32             # candidates per item whose silence is measured (the closest by cosine)


def cosine_rank(item_reps, cand_reps, exclude: Optional[Sequence[Sequence[int]]] = None,
                top_k: Optional[int] = None, chunk: int = 256) -> List[List[int]]:
    """Per item, candidate indices ordered by DESCENDING cosine similarity of their representations (cosines rounded
    to 12 decimals; ties: lower index first); exclude[i] lists candidates never offered to item i (itself, its group).
    top_k keeps only the first top_k per item. The similarity matrix is built `chunk` items at a time, so memory is
    chunk x n_candidates. Representations are the mean-pooled final-layer residuals (prepare.py)."""
    import numpy as np
    a = np.asarray(item_reps, dtype=float)
    b = np.asarray(cand_reps, dtype=float)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    idx = np.arange(b.shape[0])
    out = []
    for s0 in range(0, a.shape[0], chunk):
        sim = np.round(a[s0:s0 + chunk] @ b.T, 12) + 0.0              # + 0.0: -0.0 ties with 0.0
        for k in range(sim.shape[0]):
            i = s0 + k
            ex = set(exclude[i]) if exclude is not None else set()
            order = np.lexsort((idx, -sim[k]))
            if top_k is not None:
                order = order[:int(top_k) + len(ex)]
            r = [int(j) for j in order if int(j) not in ex]
            out.append(r[:int(top_k)] if top_k is not None else r)
    return out


def is_silent(metric_value: float, thr: float = SILENT_MAX) -> bool:
    """Daniel's latent-contrast rule ("verified silent") transferred to an output endpoint: on the candidate prompt
    the model does NOT prefer the item's good answer over its bad one, i.e. metric(good, bad | candidate) <= 0 (the
    summed log-prob difference for multi-token answers)."""
    return float(metric_value) <= thr


def build_pool(ranked: Sequence[Sequence[int]], silent: Mapping[Tuple[int, int], bool], k: int = POOL_SIZE,
               item_order: Optional[Sequence[int]] = None) -> List[int]:
    """The task-level contrast pool: round robin over items (item_order, default 0..n-1), each item contributing its
    CLOSEST not-yet-pooled candidate that is silent for it (silent[(item, candidate)]; a candidate whose silence was
    not measured is skipped), until k distinct candidates or no item has one left. Deterministic."""
    order = list(item_order) if item_order is not None else list(range(len(ranked)))
    ptr = {i: 0 for i in order}
    pool, seen = [], set()
    while len(pool) < k:
        progressed = False
        for i in order:
            r = ranked[i]
            while ptr[i] < len(r):
                j = r[ptr[i]]
                ptr[i] += 1
                if j in seen or not silent.get((i, j), False):
                    continue
                pool.append(j)
                seen.add(j)
                progressed = True
                break
            if len(pool) >= k:
                break
        if not progressed:
            break
    return pool


# ------------------------------------------------------------------------------------------- run config
CONFIG_KEYS = ("arms", "steps", "logit_norm")


def load_run_config(path: Path) -> Dict:
    """{"arms": {"W": {"lams", "lr"}, "U": {...}}, "steps", "logit_norm", "n_draws"}. Refuses a missing file and an
    arm without a lambda grid or lr."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "%s does not exist: run calibrate.py, read calib/report.txt and calib/recommendation.json, then adopt "
            "it with ADOPT=1 calibrate.py (or write the file by hand). The main run never picks its lambda grid "
            "itself." % path)
    cfg = json.loads(path.read_text())
    for k in CONFIG_KEYS:
        if k not in cfg:
            raise ValueError("%s: missing %r" % (path, k))
    for arm in FIT_ARMS:
        a = cfg["arms"].get(arm)
        if not a or not a.get("lams") or not a.get("lr"):
            raise ValueError("%s: arm %s needs a non-empty 'lams' and an 'lr'" % (path, arm))
    for arm in cfg["arms"]:
        if arm not in FIT_ARMS:
            raise ValueError("%s: unknown fitted arm %r (RND and W_ZA run at W's lambdas)" % (path, arm))
    cfg.setdefault("n_draws", N_DRAWS)
    return cfg


def prod_lr_from_080(path: Path = EXP / "080-latent-vs-output" / "config_main.json") -> Optional[Dict[str, float]]:
    """080's adopted lr per output arm, if its calibration has been adopted."""
    p = Path(path)
    if not p.exists():
        return None
    cfg = json.loads(p.read_text())
    return {a: float(v["lr"]) for a, v in cfg.get("arms", {}).items() if v.get("lr")}


# ------------------------------------------------------------------------------------------- calibration
FAIL_FAITH = 0.05
GRID_LEN = 4
COVER = (0.5, 1.0)


def fit_failed(row: Mapping) -> bool:
    if row.get("error") or row.get("skip"):
        return True
    v = num(row.get("%s_free0" % OBJ))
    return v is None or v < FAIL_FAITH


def calib_summary(rows: Sequence[Mapping]) -> Dict:
    """Per (arm, lr): failure rate, monotonicity of size and task Z in lambda, and per lambda the median size,
    median task Z and pass rate over the calibration tasks."""
    import numpy as np
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], float(r["lr"]))].append(r)
    out = {}
    for (arm, lr), rs in sorted(by.items()):
        per_t = defaultdict(list)
        for r in rs:
            per_t[r["task"]].append(r)
        mono_n, mono_f = [], []
        for t, trs in per_t.items():
            ok = [r for r in trs if not (r.get("error") or r.get("skip"))]
            lams = [float(r["lam_value"]) for r in ok]
            mono_n.append(monotone(lams, [r.get("n") for r in ok]))
            mono_f.append(monotone(lams, [r.get("%s_free0" % PRIMARY) for r in ok], tol=C80.MONO_TOL))
        lam_stats = {}
        for lam in sorted({float(r["lam_value"]) for r in rs}):
            lr_ = [r for r in rs if float(r["lam_value"]) == lam]
            ns = [num(r.get("n")) for r in lr_ if num(r.get("n")) is not None]
            fz = [num(r.get("%s_free0" % PRIMARY)) for r in lr_ if num(r.get("%s_free0" % PRIMARY)) is not None]
            lam_stats[lam_tag(lam)] = dict(n_fits=len(lr_), fail=float(np.mean([fit_failed(r) for r in lr_])),
                                           n_median=float(np.median(ns)) if ns else None,
                                           Z_median=float(np.median(fz)) if fz else None,
                                           pass_rate=float(np.mean([bool(r.get("%s_pass" % PRIMARY)) for r in lr_])))
        frac = lambda xs: (float(np.mean([bool(x) for x in xs if x is not None]))     # noqa: E731
                           if any(x is not None for x in xs) else None)
        out["%s|%g" % (arm, lr)] = dict(arm=arm, lr=lr, n_fits=len(rs), n_tasks=len(per_t),
                                        fail_rate=float(np.mean([fit_failed(r) for r in rs])),
                                        mono_size=frac(mono_n), mono_faith=frac(mono_f), lams=lam_stats)
    return out


def choose_window(lam_z: Mapping[float, Optional[float]], k: int = GRID_LEN, cover=COVER,
                  thr: float = BAND_LO) -> Tuple[List[float], bool]:
    """k consecutive swept lambdas whose median task Z best covers [cover[0], cover[1]] (ties: larger lambdas).
    Returns (lambdas ascending, brackets): the window has a lambda with median Z >= thr and one below it."""
    pts = sorted((float(l), num(z)) for l, z in lam_z.items() if num(z) is not None)
    if not pts:
        return [], False
    k = min(k, len(pts))
    best = None
    for s in range(len(pts) - k + 1):
        w = pts[s:s + k]
        zs = [z for _, z in w]
        lo, hi = min(zs), max(zs)
        cov = max(0.0, min(hi, cover[1]) - max(lo, cover[0])) / (cover[1] - cover[0])
        key = (-cov, -s)
        if best is None or key < best[0]:
            best = (key, [l for l, _ in w], max(zs) >= thr and min(zs) < thr)
    return sorted(best[1]), best[2]


def recommend(summary: Mapping, prod_lr: float = PROD_LR, k: int = GRID_LEN) -> Dict:
    """Per fitted arm: the lr with the lowest failure rate (ties: more faithfulness-monotone, then nearest 0.05),
    then choose_window. Warnings when the window does not bracket Z = 0.8 or failures exceed 25%."""
    rec = {"arms": {}, "warnings": []}
    for arm in sorted({v["arm"] for v in summary.values()}):
        cands = [v for v in summary.values() if v["arm"] == arm]
        cands.sort(key=lambda v: (v["fail_rate"], -(v["mono_faith"] or 0.0), abs(math.log10(v["lr"] / prod_lr))))
        best = cands[0]
        zmed = {float(l): s["Z_median"] for l, s in best["lams"].items()}
        grid, brackets = choose_window(zmed, k)
        rec["arms"][arm] = dict(lr=best["lr"], lams=grid, fail_rate=best["fail_rate"], mono_faith=best["mono_faith"],
                                mono_size=best["mono_size"], brackets=brackets)
        if not brackets:
            rec["warnings"].append("%s: no %d-lambda window of the swept grid brackets median task Z = %.1f; extend "
                                   "the sweep before adopting" % (arm, k, BAND_LO))
        if best["fail_rate"] > 0.25:
            rec["warnings"].append("%s: failure rate %.0f%% at the best lr" % (arm, 100 * best["fail_rate"]))
    return rec


# ------------------------------------------------------------------------------------------- curves (claims 1-2)
def passes(row: Mapping, prefix: str = PRIMARY, rule: str = "ZAC") -> bool:
    """The v1 rule on a task metric: Z, A_sp, C_sp in [0.8, 1.5] and necessity >= 0.9 (rule "ZAC"); rule "ZA" drops
    C (the sensitivity reading for generated-contrast tasks)."""
    fs = FAITH if rule == "ZAC" else FAITH[:2]
    vals = [num(row.get("%s_%s" % (prefix, f))) for f in fs]
    nec = num(row.get("%s_nec" % prefix))
    return all(v is not None and BAND_LO <= v <= BAND_HI for v in vals) and nec is not None and nec >= NEC_MIN


def curve(rows: Sequence[Mapping], prefix: str = PRIMARY) -> List[Dict]:
    """One arm's size-faithfulness curve for one task: [{lam, n, Z, A, C, nec, passed, passed_ZA, acc, Z_a1}]."""
    out = []
    for r in rows:
        if r.get("error") or r.get("skip") or num(r.get("n")) is None:
            continue
        out.append(dict(lam=float(r["lam_value"]), n=int(r["n"]), Z=num(r.get("%s_free0" % prefix)),
                        A=num(r.get("%s_freeM_topk" % prefix)), C=num(r.get("%s_freeN_topk" % prefix)),
                        nec=num(r.get("%s_nec" % prefix)), passed=passes(r, prefix),
                        passed_ZA=passes(r, prefix, "ZA"), acc=num(r.get("%s_acc_free0" % prefix)),
                        Z_a1=num(r.get("%s_free0_a1" % prefix))))
    return sorted(out, key=lambda p: (p["n"], -p["lam"]))


def smallest_passing(cv: Sequence[Mapping]) -> Optional[int]:
    ns = [p["n"] for p in cv if p["passed"] and p["n"] > 0]
    return min(ns) if ns else None


def size_ratio(cw: Sequence[Mapping], cu: Sequence[Mapping], thr: float = BAND_LO) -> Dict:
    """Claim 2 per task: size to reach task Z >= thr per arm and U / W; smallest passing size per arm and its
    ratio. U "never" with W reached counts as +inf."""
    nw, sw = size_to_reach([p["n"] for p in cw], [p["Z"] for p in cw], thr)
    nu, su = size_to_reach([p["n"] for p in cu], [p["Z"] for p in cu], thr)
    ratio = None
    if nw:
        ratio = (nu / nw) if nu else float("inf")
    pw, pu = smallest_passing(cw), smallest_passing(cu)
    pratio = None
    if pw:
        pratio = (pu / pw) if pu else float("inf")
    return dict(n_reach_W=nw, status_W=sw, n_reach_U=nu, status_U=su, reach_ratio=ratio, pass_W=pw, pass_U=pu,
                pass_ratio=pratio, u_max_n=max([p["n"] for p in cu], default=None))


def u_at_w_sizes(cw: Sequence[Mapping], cu: Sequence[Mapping]) -> List[Dict]:
    """At every W circuit's size: U's task Z, A, C, necessity and pass indicator interpolated in log10 size."""
    out = []
    for p in cw:
        row = dict(lam=p["lam"], n=p["n"], W_Z=p["Z"], W_pass=p["passed"])
        for f in ("Z", "A", "C", "nec"):
            y, ext = interp_log_n([q["n"] for q in cu], [q[f] for q in cu], p["n"])
            row["U_" + f], row["U_extrap"] = y, ext
        y, _ = interp_log_n([q["n"] for q in cu], [1.0 if q["passed"] else 0.0 for q in cu], p["n"])
        row["U_pass"] = y
        out.append(row)
    return out


def primary_lambda(cw: Sequence[Mapping]) -> Tuple[Optional[float], bool]:
    """The W circuit the structure and T5 analyses use: the smallest passing circuit; if none passes, the best
    worst-ablation one (max over lambda of min(Z, A, C)), flagged not passing."""
    ok = [p for p in cw if p["passed"]]
    if ok:
        return min(ok, key=lambda p: p["n"])["lam"], True
    scored = [p for p in cw if all(p[f] is not None for f in ("Z", "A", "C"))]
    if not scored:
        return None, False
    return max(scored, key=lambda p: min(p["Z"], p["A"], p["C"]))["lam"], False


# ------------------------------------------------------------------------------------------- random null
def draw_support(w_counts: Mapping[Tuple[int, str], int], active: Mapping[Tuple[int, str], Sequence[int]],
                 rng: random.Random) -> Tuple[Dict[Tuple[int, str], List[int]], int]:
    """A random circuit matched to W's per-site member counts, drawn uniformly from the latents active on the task's
    training prompts at that site (062 random_null.py's recipe). Returns (support, number of short sites)."""
    sup, short = {}, 0
    for s in sorted(w_counts):
        pool = sorted(int(i) for i in active.get(s, ()))
        n = int(w_counts[s])
        if len(pool) < n:
            short += 1
        sup[s] = rng.sample(pool, min(n, len(pool)))
    return sup, short


def site_counts(alphas: Mapping) -> Dict[Tuple[int, str], int]:
    return {s: len(d) for s, d in alphas.items() if d}


def alphas_to_json(alphas: Mapping) -> Dict[str, Dict[str, float]]:
    return {"%d/%s" % s: {str(i): float(a) for i, a in sorted(d.items())} for s, d in sorted(alphas.items()) if d}


def alphas_from_json(d: Mapping) -> Dict[Tuple[int, str], Dict[int, float]]:
    out = {}
    for k, v in d.items():
        l, kind = k.split("/")
        out[(int(l), kind)] = {int(i): float(a) for i, a in v.items()}
    return out


def member_set(alphas: Mapping) -> set:
    return {(int(s[0]), str(s[1]), int(i)) for s, d in alphas.items() for i in d}


def member_key(m: Tuple[int, str, int]) -> str:
    return "%d.%s.%d" % m


# ------------------------------------------------------------------------------------------- structure (T4)
CONC = 0.6                    # a member is "concentrated" on a role when >= 60% of its activation mass sits there
ANCHOR = "anchor"             # the prediction position is its own bucket, whatever its template role
ANSWER = "answer"             # teacher-forced answer positions (multi-token answers)
B_STRUCT = 2000
ENRICH_MIN = 2.0              # a published head set is "found" when its share is >= 2x its uniform base rate
N_HEADS = 12                  # GPT-2 small

# Published GPT-2 small circuits (layer, head). The positions are where the papers place each class.
IOI_HEADS = {
    "name_mover": dict(heads=((9, 6), (9, 9), (10, 0)), bucket=ANCHOR),
    "s_inhibition": dict(heads=((7, 3), (7, 9), (8, 6), (8, 10)), bucket=ANCHOR),
    "induction": dict(heads=((5, 5), (6, 9)), bucket="s2"),
    "duplicate_token": dict(heads=((0, 1), (3, 0)), bucket="s2"),
    "previous_token": dict(heads=((2, 2), (4, 11)), bucket="s1_next"),
}
GT_HEADS = dict(heads=((5, 1), (5, 5), (6, 1), (6, 9), (7, 10), (8, 8), (8, 11), (9, 1)), bucket=ANCHOR)
GT_MLP_LAYERS = (8, 9, 10, 11)
GT_YY_ROLES = ("y1_yy",)


def dominant_role(act: Mapping[str, float], conc: float = CONC) -> Optional[str]:
    tot = sum(max(0.0, float(v)) for v in act.values())
    if tot <= 0:
        return None
    r, v = max(act.items(), key=lambda kv: kv[1])
    return r if float(v) / tot >= conc else None


def _median_diff_ci(a: Sequence[float], b: Sequence[float], B: int = B_STRUCT, seed: int = TASK_SEED):
    """median(a) - median(b) with a 95% bootstrap interval (members resampled within each group)."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    d = float(np.median(a) - np.median(b))
    bs = [np.median(a[rng.integers(0, len(a), len(a))]) - np.median(b[rng.integers(0, len(b), len(b))])
          for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d, float(lo), float(hi)


def role_contrib(profile: Sequence[Mapping]) -> Dict[str, float]:
    """Total |attribution| per bucket over the circuit's members."""
    out: Dict[str, float] = defaultdict(float)
    for m in profile:
        for r, v in (m.get("contrib") or {}).items():
            out[r] += abs(float(v))
    return dict(out)


def layers_concentrated(profile: Sequence[Mapping], roles: Iterable[str]) -> List[int]:
    roles = set(roles)
    return [int(m["layer"]) for m in profile if dominant_role(m.get("act") or {}) in roles]


def head_weights(profile: Sequence[Mapping], bucket: str) -> Dict[Tuple[int, int], float]:
    """Per (layer, head): the attention members' |attribution| at `bucket`, distributed over heads in proportion to
    each member's |head drive| there (engine.head_drive: the exact per-head decomposition of an attn-out latent's
    pre-activation). Only attn members with head data contribute."""
    out: Dict[Tuple[int, int], float] = defaultdict(float)
    for m in profile:
        hd = (m.get("heads") or {}).get(bucket)
        w = abs(float((m.get("contrib") or {}).get(bucket, 0.0)))
        if not hd or w == 0:
            continue
        tot = sum(abs(float(v)) for v in hd.values())
        if tot <= 0:
            continue
        for h, v in hd.items():
            out[(int(m["layer"]), int(h))] += w * abs(float(v)) / tot
    return dict(out)


def head_enrichment(weights: Mapping[Tuple[int, int], float], heads: Sequence[Tuple[int, int]],
                    n_heads: int = N_HEADS) -> Dict:
    """Share of the head weight in the published heads' LAYERS that sits on the published heads, against the
    uniform base rate (published heads / all heads in those layers). enrichment = share / base."""
    layers = sorted({l for l, _ in heads})
    tot = sum(v for (l, _), v in weights.items() if l in layers)
    hit = sum(weights.get(h, 0.0) for h in heads)
    base = len(heads) / (n_heads * len(layers)) if layers else None
    share = hit / tot if tot > 0 else None
    return dict(share=share, base=base, enrichment=(share / base) if (share is not None and base) else None,
                weight_in_layers=tot)


def structure_checks(family: str, task: str, profile: Sequence[Mapping]) -> Dict:
    """The registered T4 checks on one task circuit's member profile (README, T4). Each: value, held (True / False),
    or None when the check cannot be evaluated (no members of that class). Primary: IOI and greater-than (published
    GPT-2 small circuits). Secondary: agreement (SFC's circuits are Pythia / Gemma circuits, not GPT-2's)."""
    out: Dict[str, Dict] = {}
    rc = role_contrib(profile)
    total = sum(rc.values()) or float("nan")
    if family == "ioi":
        for cls, spec in IOI_HEADS.items():
            e = head_enrichment(head_weights(profile, spec["bucket"]), spec["heads"])
            out["ioi_" + cls] = dict(desc="%s heads %s at %s: share of attn-member attribution in their layers >= "
                                          "%gx the uniform base rate" % (cls, spec["heads"], spec["bucket"], ENRICH_MIN),
                                     value=e["enrichment"], share=e["share"], base=e["base"],
                                     held=None if e["enrichment"] is None else bool(e["enrichment"] >= ENRICH_MIN),
                                     primary=True)
    elif family == "greater_than":
        e = head_enrichment(head_weights(profile, GT_HEADS["bucket"]), GT_HEADS["heads"])
        out["gt_heads"] = dict(desc="heads %s at the prediction position: share >= %gx the uniform base rate"
                                    % (GT_HEADS["heads"], ENRICH_MIN), value=e["enrichment"], share=e["share"],
                               base=e["base"], held=None if e["enrichment"] is None else bool(e["enrichment"] >= ENRICH_MIN),
                               primary=True)
        mlp = [m for m in profile if m["kind"] == "mlp"]
        tot_mlp = sum(abs(float((m.get("contrib") or {}).get(ANCHOR, 0.0))) for m in mlp)
        late = sum(abs(float((m.get("contrib") or {}).get(ANCHOR, 0.0))) for m in mlp if m["layer"] in GT_MLP_LAYERS)
        share = late / tot_mlp if tot_mlp > 0 else None
        out["gt_late_mlps"] = dict(desc="MLP members in layers 8-11 carry >= 50%% of the MLP members' |attribution| at "
                                        "the prediction position (uniform base rate 4/12)", value=share,
                                   held=None if share is None else bool(share >= 0.5), primary=True)
        d, lo, hi = _median_diff_ci(layers_concentrated(profile, GT_YY_ROLES), layers_concentrated(profile, [ANCHOR]))
        out["gt_layer_order"] = dict(desc="median layer of YY-concentrated members < anchor-concentrated (CI below 0)",
                                     value=d, lo=lo, hi=hi, held=None if d is None else bool(hi < 0), primary=True)
    elif family == "sva":
        variant = task.replace("sva_", "")
        if variant in ("across_rc", "across_pp"):
            d, lo, hi = _median_diff_ci(layers_concentrated(profile, ["subj"]), layers_concentrated(profile, [ANCHOR]))
            out["sva_layer_order"] = dict(desc="median layer of subject-concentrated members < anchor-concentrated "
                                               "(CI below 0)", value=d, lo=lo, hi=hi,
                                          held=None if d is None else bool(hi < 0), primary=False)
        if variant in ("within_rc", "across_rc"):
            s = rc.get("subj", 0.0) + (rc.get(ANCHOR, 0.0) if variant == "within_rc" else 0.0)
            dd = rc.get("distractor", 0.0)
            out["sva_subject_over_distractor"] = dict(desc="|attribution| on the agreeing noun > on the distractor",
                                                      value=s / dd if dd else float("inf"), held=bool(s > dd),
                                                      primary=False)
        if variant != "simple":
            sh = (rc.get("subj", 0.0) + (rc.get(ANCHOR, 0.0) if variant == "within_rc" else 0.0)) / total
            out["sva_subject_share"] = dict(desc="share of |attribution| on the agreeing noun >= 0.10", value=sh,
                                            held=bool(sh >= 0.10), primary=False)
    return out


# ------------------------------------------------------------------------------------------- link to 062 (T5)
def link_to_latent_circuits(top: Sequence[Tuple[int, str, int]], task_members: set, targets: set,
                            latent_circuits: Mapping[str, set], active_pool: Optional[set] = None) -> Dict:
    """T5 (descriptive): how many of the top members are production targets, the union of their latent circuits and
    its overlap with the task circuit, with base rates."""
    keys = [member_key(m) for m in top]
    hit = [k for k in keys if k in targets]
    union = set()
    for k in hit:
        union |= set(latent_circuits.get(k, ()))
    ov = overlap(task_members, union) if union else dict(jaccard=None, cont_a_in_b=None, cont_b_in_a=None, n_inter=0)
    base_members = (sum(1 for m in task_members if member_key(m) in targets) / len(task_members)
                    if task_members else None)
    base_active = (sum(1 for m in active_pool if member_key(m) in targets) / len(active_pool)
                   if active_pool else None)
    return dict(n_top=len(keys), n_top_targets=len(hit), top_targets=hit, n_union=len(union),
                n_task=len(task_members), task_in_union=ov["cont_a_in_b"], union_in_task=ov["cont_b_in_a"],
                jaccard=ov["jaccard"], n_inter=ov["n_inter"], base_rate_members=base_members,
                base_rate_active=base_active, n_latent_circuits_found=sum(1 for k in hit if k in latent_circuits))


# ------------------------------------------------------------------------------------------- verdicts
def verdict_t1(passed_any: Mapping[str, bool]) -> str:
    if not passed_any:
        return "NO DATA"
    k = sum(bool(v) for v in passed_any.values())
    return "SUPPORTED" if k == len(passed_any) else ("MIXED" if 2 * k >= len(passed_any) else "NOT SUPPORTED")


def verdict_t2(ratios: Mapping[str, Mapping]) -> Tuple[str, Dict]:
    import numpy as np
    rr = [v["reach_ratio"] for v in ratios.values() if v.get("reach_ratio") is not None]
    at = [v["u_fails_at_w_pass"] for v in ratios.values() if v.get("u_fails_at_w_pass") is not None]
    if not rr:
        return "NO DATA", dict(n=0)
    share2 = float(np.mean([r >= 2 for r in rr]))
    med = float(np.median(rr))
    share_fail = float(np.mean(at)) if at else None
    ok = share2 >= 0.75 and med >= 2 and (share_fail is None or share_fail >= 0.75)
    return ("SUPPORTED" if ok else "NOT SUPPORTED"), dict(n=len(rr), share_ratio_ge2=share2, median_ratio=med,
                                                          share_u_fails_at_w_pass=share_fail, n_at=len(at))


def verdict_t3(rnd_pass: Sequence[bool], rnd_below_w: Mapping[str, bool]) -> Tuple[str, Dict]:
    if not rnd_pass:
        return "NO DATA", dict(n=0)
    rate = sum(bool(x) for x in rnd_pass) / len(rnd_pass)
    below = all(rnd_below_w.values()) if rnd_below_w else None
    ok = rate <= 0.05 and below is not False
    return ("SUPPORTED" if ok else "NOT SUPPORTED"), dict(n=len(rnd_pass), pass_rate=rate,
                                                          upper95=clopper_pearson_upper(sum(map(bool, rnd_pass)),
                                                                                        len(rnd_pass)),
                                                          all_tasks_below_w=below)


def verdict_t4(checks: Mapping[str, Mapping[str, Mapping]], primary_only: bool = True) -> Tuple[str, Dict]:
    """T4 over the PRIMARY checks (IOI, greater-than): every evaluable check holds -> SUPPORTED; at least half ->
    MIXED; otherwise NOT SUPPORTED. primary_only=False also counts the secondary (agreement) checks."""
    held = [c["held"] for t in checks.values() for c in t.values()
            if c.get("held") is not None and (c.get("primary", True) or not primary_only)]
    if not held:
        return "NO DATA", dict(n=0)
    k = sum(bool(h) for h in held)
    v = "SUPPORTED" if k == len(held) else ("MIXED" if 2 * k >= len(held) else "NOT SUPPORTED")
    return v, dict(n=len(held), n_held=k)


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> Optional[float]:
    """Exact upper bound (alpha/2 tail) of a binomial proportion."""
    if n == 0:
        return None
    if k >= n:
        return 1.0

    def cdf(p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha / 2:
            lo = mid
        else:
            hi = mid
    return hi
