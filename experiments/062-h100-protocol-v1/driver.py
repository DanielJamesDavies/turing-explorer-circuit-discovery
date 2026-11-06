"""PROTOCOL-V1 H100 DRIVER: fits and scores weighted circuits under the final protocol, one GPU per process.

Protocol v1 (DAN-75 spec, 2026-09-25):
  - contexts   32 strongest + 16 mid-band training contexts, stratified rotating split (ranks 1-2 always train),
               held-out 16 strongest (primary) + 16 mid-band (reported); 64 close contrast contexts, verified
               silent, stratified by similarity (48 train / 16 held-out). Built per target with 059's builder
               (the run); now src/circuit/protocol_contexts.py, which reproduces it exactly (DAN-78).
               Thin targets (no mid-band pool) fall back to the 48 strongest, flagged thin.
  - method     WCM[Z+C+A], gamma_C = gamma_A = 0.25, lambda = 1e-3, rank-keep 3e-3, ablation values from the
               training contexts, AdamW lr 0.05 / wd 0.05, 400 steps, batch 4.
The contexts reach the production engine through the same injection as 059 / 061 (protocol_harness), which
reproduced 059 arm B exactly. Production discovery can now take them without injection:
config.discovery.context_protocol = "v1" (DAN-78; check_protocol_contexts.py verifies the equivalence).

MODE=main   per target: contexts -> fit -> eval pass (held-out strongest and mid-band) -> specificity (056)
MODE=sweep  the Figure 4 / DAN-76 data: per arm (WCM and unweighted masking over a lambda grid, gamma 0.25)
            fit + eval on held-out strongest. WCM at lambda 1e-3 reuses the main circuit when it exists.

Resumable at every step (one file per target for contexts and circuits; jsonl rows keyed by target / arm / held).
Each process takes targets[i::k] for SHARD=i/k.

  CUDA_VISIBLE_DEVICES=0 SHARD=0/8 MODE=main PYTHONPATH=src python experiments/062-h100-protocol-v1/driver.py
  env: MODE (main | sweep)  SHARD (0/1)  TARGETS (default targets_stage1.txt / targets_sweep.txt)
       OUT (default experiments/062-h100-protocol-v1/out)  LIMIT (first N of this shard, smoke tests)
       SWEEP_ARMS (comma list, default all)  SKIP_SPEC=1  CURVE_STRIDE (1 = every step of the training curve)
       ORDER (interleaved | list)  SLOT (co-tenant slot on the GPU; launch.sh sets it)
       LAM (the main fit's sparsity penalty, 1e-3)  STEPS (the training budget, 400; smoke tests only)
       FAMILY (turingllm | pythia-70m | gpt2-small)  DATA / STORES (public families: the shard directory and the
       data pass's artefact root, default data_public/<FAMILY> / outputs_public/<FAMILY>)  SKIP_MISSING_SAES=1
       CTX_DIR (read the context records from here, e.g. 086's rebuilt out_full/out/ctx_v2; forces EVAL_SPLIT=explicit)
       (the only rebuilt variant: the source-file rule and ctx_v2_nosrc were dropped 2026-10-24)
       GROUP_SPLIT (public families: 1 = group-aware split + held-out rule, the default; 0 = the per-row split)
       WCM_VERSION (v1 = protocol v1, the default | v2 = A / C trained under the scorer's sparsity-preserving fill,
       straight-through, against the scorer's means | v2cr = v2 + the donor-resample term R_C, 095's v2CR | v3 = v2cr
       with R weight 1.0 and rank-keep 1e-1 (2026-11-01, exp 102) | v4 = v3 + geometric lr x3.15 with a 0.1 warm-up +
       the censored-activation term, the production algorithm from 2026-11-04 (exp 107 latent_pilot) | v4nr = v4
       without R; see WCM_VERSIONS / WCM_LR_MULT / WCM_RANK). Rows carry wcm_version; OUT is version-marked.
       R_WEIGHT / R_POOL / R_K (v2cr only)

Every fresh fit also writes train.shardN.jsonl: fit stats + the per-step training curve (see train_row).

FAMILY (DAN-40/41). turingllm (default) is the 15k run's path, unchanged: 049 amp_eval_pass_v2.setup() (TuringLLM's
outputs/ stores) and 049's scorer through a patched M0 (EVAL_SPLIT, default as_run). A public family (069 data pass +
070 adapters) gets the same G keys from the family's Inference / SAEBank and the stores under STORES (069
common.configure / load_stores; configured at import, before anything imports store.*): K = bank.k, kinds from the
bank (GPT-2: attn / resid_mid / mlp / resid). It fits through the same Runner.method / Runner.fit, scores with
src eval/wcm_eval.score_circuit(split="explicit") on circuit/protocol_contexts.eval_probe, specificity with
eval/wcm_specificity.specificity_rows, and runs float32 matmuls at "high" (TF32), as the 068 gate fits did.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).parent
EXP = HERE.parent
REPO = EXP.parent
for d in ("059-context-pool", "049-circuit-graph", "056-specificity"):
    sys.path.insert(0, str(EXP / d))
MODE = os.environ.get("MODE", "main")
SHARD_I, SHARD_K = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
TARGETS = Path(os.environ.get("TARGETS", str(HERE / ("targets_stage1.txt" if MODE == "main" else "targets_sweep.txt"))))
OUT = Path(os.environ.get("OUT", str(HERE / "out")))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
SKIP_SPEC = os.environ.get("SKIP_SPEC") == "1"
GAMMA, LAM = 0.25, float(os.environ.get("LAM", "1e-3"))           # LAM: the main fit's price (DAN-42 pilot sets it)
W_LAMS = (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3)
U_LAMS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
STEPS = int(os.environ.get("STEPS", "400"))                          # the protocol's training budget (env: smoke only)
# WCM_VERSION (2026-10-25): the WCM protocol the fits run under. v1 (default) = protocol v1, unchanged. v2 = the A and
# C training terms under the pass rule's sparsity-preserving fill with straight-through gradients, against the
# scorer's own A / C means (src config.discovery.learned_mask train_fill / sparse_grad / train_means_source; experiments
# 090-093); everything else as v1. Every output row carries wcm_version. Runner refuses an OUT whose circuits were
# fitted under another version (OUT/WCM_VERSION marker; a v2 run never reuses unmarked, i.e. v1, circuits).
# v2cr (2026-10-31, Daniel approved for the pod; experiment 101) = v2 + R_C: the donor-resample term of 095's v2CR arm
# (Z + A + C + R_C), native in src (learned_mask resample_*): gamma_R 0.25, donors = the target's 48 training contrast
# contexts (eval_reference's C rows), one per training row redrawn every forward (seed 20261017), under the
# sparsity-preserving donor fill. Env overrides, v2cr only: R_WEIGHT (gamma_R), R_POOL (contrast | activating = the 48
# strongest-train rows), R_K (donor draws per row per step). The R settings are marked in OUT/WCM_R next to OUT/WCM_VERSION
# (a run with other R settings refuses that OUT). v1 / v2 set resample_weight 0 explicitly (the config default: unchanged).
# v3 (2026-11-01, Daniel approved as the final production algorithm; experiments 102, 104) = v2cr with gamma_R 1.0 and
# rank-keep 1e-1 (102's arm r1.0_rk1e-1): every arm of a v3 run (main, every W_ / U_ sweep arm, the companions) takes
# both. Fixed: no R_* overrides. The rank-keep weight is per version (WCM_RANK); v1 / v2 / v2cr keep 3e-3 (unchanged).
_NO_R = dict(resample_weight=0.0, resample_pool="contrast", resample_k=1)
WCM_VERSIONS = {
    "v1": dict(train_fill="dense", sparse_grad="ste", train_means_source="fit", **_NO_R),
    "v2": dict(train_fill="sparse", sparse_grad="ste", train_means_source="eval_ref", **_NO_R),
    "v2cr": dict(train_fill="sparse", sparse_grad="ste", train_means_source="eval_ref",
                 resample_weight=0.25, resample_pool="contrast", resample_k=1),
    "v3": dict(train_fill="sparse", sparse_grad="ste", train_means_source="eval_ref",
               resample_weight=1.0, resample_pool="contrast", resample_k=1),
}
# v4 (2026-11-04, Daniel approved for the pod; experiment 107 latent_pilot, dev48 + the 102 check panel + GPT-2 /
# Pythia) = v3 + the geometric lr schedule (lr * 0.05**progress, the gate anneal's own shape: lr / T fixed) at peak
# x3.15 (restores constant's sum(lr), the pruning budget) with a 0.1 warm-up (the x3.15 peak's early spikes), + the
# censored-activation data term (censor_temp 0.05: the A / C terms train the post-Top-K activation the scorer reads).
# v4nr = v4 without R (T10's without-R arm: the pair differs in R alone). Fixed: no LR_* / CENSOR_T overrides.
WCM_VERSIONS["v4"] = dict(WCM_VERSIONS["v3"], lr_schedule="geometric", warmup_frac=0.1, censor_temp=0.05)
WCM_VERSIONS["v4nr"] = dict(WCM_VERSIONS["v4"], **_NO_R)
WCM_LR_MULT = {"v4": 3.15, "v4nr": 3.15}                           # the version's peak-lr multiplier (default 1)
WCM_RANK = {"v1": 3e-3, "v2": 3e-3, "v2cr": 3e-3, "v3": 1e-1,       # the version's rank-keep weight (every arm)
            "v4": 1e-1, "v4nr": 1e-1}
WCM_VERSION = os.environ.get("WCM_VERSION", "v1")
if WCM_VERSION not in WCM_VERSIONS:
    raise SystemExit("WCM_VERSION must be one of %s, got %r" % (sorted(WCM_VERSIONS), WCM_VERSION))
_R_ENV = {k: os.environ[k] for k in ("R_WEIGHT", "R_POOL", "R_K") if os.environ.get(k)}
if _R_ENV and WCM_VERSION != "v2cr":
    raise SystemExit("R_WEIGHT / R_POOL / R_K apply to WCM_VERSION=v2cr only (got %s with %s)" % (WCM_VERSION, _R_ENV))
if _R_ENV:
    WCM_VERSIONS["v2cr"].update(resample_weight=float(_R_ENV.get("R_WEIGHT", 0.25)),
                                resample_pool=_R_ENV.get("R_POOL", "contrast"), resample_k=int(_R_ENV.get("R_K", 1)))
    if WCM_VERSIONS["v2cr"]["resample_pool"] not in ("contrast", "activating") or WCM_VERSIONS["v2cr"]["resample_k"] < 1 \
            or WCM_VERSIONS["v2cr"]["resample_weight"] <= 0:
        raise SystemExit("bad R override %s" % _R_ENV)
# The protocol's rank-keep weight under this process's version: the sweep grid's default and the reference for the
# arm-name suffix _r<weight> (so W_0.001 means the version's own settings; W_0.001_r0.003 under v3 is the old weight).
RANK = WCM_RANK[WCM_VERSION]
# LR_SCHEDULE (2026-11-02, exp 107 arena C §8; pilot only): override the learned mask's lr schedule (e.g. "geometric" =
# lr * 0.05**progress, the gate anneal's own shape). Unset = the config's (constant), unchanged. An OUT fitted under an
# override is marked OUT/LR_SCHEDULE and refuses to mix with another schedule's circuits (check_out_version).
LR_SCHEDULE = os.environ.get("LR_SCHEDULE") or None
# LR_PEAK_MULT (pilot only): multiply the learned mask's peak lr. A decaying schedule spends less lr in total (geometric
# 0.05: sum(lr) ~0.32x constant's), and both budgets scale with sum(lr) (sparsity: theta walks from theta_init across 0
# at ~lr per step; decay: lr * weight_decay), so ~3.15 restores constant's sum. Unset / 1 = unchanged.
LR_PEAK_MULT = float(os.environ.get("LR_PEAK_MULT") or 1.0)
_BASE_LR = []                                                       # the config's lr, read once (method() sets lm.lr)
# RANK_CUT_K (pilot only): rank term in "cut" mode against the RANK_CUT_K-th rival instead of the version's "keep"
# (hold the clean rank), at the version's rank weight. Near-threshold attention seeds (clean rank 65-117 of k 128)
# held their clean rank in training but were censored at scoring; a cut well inside the Top-K leaves a margin.
RANK_CUT_K = int(os.environ["RANK_CUT_K"]) if os.environ.get("RANK_CUT_K") else None
# CENSOR_T / RANK_ZERO_MULT / RIVAL_W (pilot only, learned_mask censor_temp / rank_zero_mult / rival_weight): the
# censored-activation data term, the rank term's weight inside the zero ablation, the rival term
CENSOR_T = float(os.environ["CENSOR_T"]) if os.environ.get("CENSOR_T") else None
RANK_ZERO_MULT = float(os.environ.get("RANK_ZERO_MULT") or 1.0)
RIVAL_W = float(os.environ.get("RIVAL_W") or 0.0)
# LAM_ADAPT_EPS / _RATE / _FRAC (pilot only, learned_mask lam_adapt_*): adaptive λ, starting at LAM
LAM_ADAPT_EPS = float(os.environ["LAM_ADAPT_EPS"]) if os.environ.get("LAM_ADAPT_EPS") else None
LAM_ADAPT_RATE = float(os.environ.get("LAM_ADAPT_RATE") or 0.02)
LAM_ADAPT_FRAC = float(os.environ.get("LAM_ADAPT_FRAC") or 0.7)
LAM_ADAPT_UP = float(os.environ.get("LAM_ADAPT_UP") or 30.0)
# R_SEED (seed replicates only): the R term's donor-draw seed (learned_mask resample_seed; the fit's only random
# source). Unset = the config's (20261017), unchanged.
R_SEED = int(os.environ["R_SEED"]) if os.environ.get("R_SEED") else None
# NEC_W / NEC_T (pilot only): the necessity term's weight and target (learned_mask necessity_*); LR_WARMUP: the lr
# warm-up share of the run (learned_mask warmup_frac, for the geometric schedule's higher peak). Unset = off.
NEC_W = float(os.environ.get("NEC_W") or 0.0)
NEC_T = float(os.environ.get("NEC_T") or 0.95)
LR_WARMUP = float(os.environ.get("LR_WARMUP") or 0.0)
LAM_ADAPT_DOWN = float(os.environ.get("LAM_ADAPT_DOWN") or 30.0)
if WCM_VERSION in WCM_LR_MULT and (LR_SCHEDULE or LR_PEAK_MULT != 1.0 or CENSOR_T or LR_WARMUP):
    raise SystemExit("LR_SCHEDULE / LR_PEAK_MULT / CENSOR_T / LR_WARMUP are fixed under WCM_VERSION=%s" % WCM_VERSION)
# what OUT/LR_SCHEDULE records: the fit overrides (None = none: unmarked)
_MARKS = ([LR_SCHEDULE or "config"] if (LR_SCHEDULE or LR_PEAK_MULT != 1.0) else []) \
    + (["lr*%g" % LR_PEAK_MULT] if LR_PEAK_MULT != 1.0 else []) + (["rankcut%d" % RANK_CUT_K] if RANK_CUT_K else []) \
    + (["censor%g" % CENSOR_T] if CENSOR_T else []) + (["rankzero*%g" % RANK_ZERO_MULT] if RANK_ZERO_MULT != 1.0 else []) \
    + (["rival%g" % RIVAL_W] if RIVAL_W else []) \
    + (["lamadapt%g/%g/%g" % (LAM_ADAPT_EPS, LAM_ADAPT_RATE, LAM_ADAPT_FRAC)
        + ("" if (LAM_ADAPT_UP, LAM_ADAPT_DOWN) == (30.0, 30.0) else "/x%g-/%g" % (LAM_ADAPT_UP, LAM_ADAPT_DOWN))]
       if LAM_ADAPT_EPS else []) \
    + (["rseed%d" % R_SEED] if R_SEED is not None else []) \
    + (["nec%g/%g" % (NEC_W, NEC_T)] if NEC_W else []) + (["warmup%g" % LR_WARMUP] if LR_WARMUP else [])
LR_MARK = " ".join(_MARKS) or None
SWEEP =([("W_%g" % l, True, l, STEPS, GAMMA, RANK) for l in W_LAMS]
         + [("U_%g" % l, False, l, STEPS, GAMMA, RANK) for l in U_LAMS])
# SWEEP_ARMS picks arms by name, W_<lambda> (WCM) or U_<lambda> (unweighted), and may name sparsity penalties outside
# the default grids (e.g. W_1e-4 extends WCM beyond ~1k nodes). Optional suffixes change one setting from the protocol:
# _s<steps> the training budget (W_0.001_s800, the DAN-131 depth test), _g<gamma> the A / C term weight, _r<weight>
# the rank-keep weight (W_0.001_g1_r0.03, the attention test). Names are normalised to the "%g" form used on disk.
WANT = []
for a in (x for x in os.environ.get("SWEEP_ARMS", "").split(",") if x):
    parts = a.split("_")
    w, lam = parts[0] == "W", float(parts[1])
    opt = {p[0]: p[1:] for p in parts[2:]}
    steps, gamma, rank = int(opt.get("s", STEPS)), float(opt.get("g", GAMMA)), float(opt.get("r", RANK))
    WANT.append("%s_%g" % (parts[0], lam) + ("_s%d" % steps if steps != STEPS else "")
                + ("_g%g" % gamma if gamma != GAMMA else "") + ("_r%g" % rank if rank != RANK else ""))
    if not any(a_[0] == WANT[-1] for a_ in SWEEP):
        SWEEP.append((WANT[-1], w, lam, steps, gamma, rank))
SWEEP = [a for a in SWEEP if not WANT or a[0] in WANT]
TAG = "shard%d" % SHARD_I
# Co-tenant slot on this process's GPU (launch.sh: shard i runs on GPU i % K, so SLOT = i // K). Drives the
# interleaved order's rotation; see interleave().
SLOT = int(os.environ.get("SLOT", "0"))
ORDER = os.environ.get("ORDER", "interleaved")          # interleaved | list (the target file's own order)
KIND_RANK = {"attn": 0, "resid_mid": 1, "mlp": 2, "resid": 3}  # within-layer order (GPT-2's resid_mid: after attn)

# Model family (see the module docstring). A public family is configured here, at import: the store singletons take
# their grid from config.model.family when store.* is first imported, and the companions (random_null, stable_core,
# 073) import src modules that import store.* before they build the Runner.
FAMILY = os.environ.get("FAMILY", "turingllm")
PUBLIC = FAMILY != "turingllm"
DATA = os.environ.get("DATA") or (str(REPO / "data_public" / FAMILY) if PUBLIC else None)
STORES = os.environ.get("STORES") or (str(REPO / "outputs_public" / FAMILY) if PUBLIC else None)
SKIP_MISSING_SAES = os.environ.get("SKIP_MISSING_SAES") == "1"   # lazy dictionaries; a missing one = a missing site
# CTX_DIR (DAN-157): read the context records from this directory instead of OUT/ctx, e.g. the rebuilt held-out sets
# (086 rebuild.py: out_full/out/ctx_v2). Read only: a missing record is an error, never built there.
# It forces the explicit eval split (EVAL_SPLIT=explicit; any other value is refused): a rebuilt held-out set may hold
# fewer than 16 rows, and the as_run split would then move reference rows into the held-out slice. A rebuilt record's
# deduplicated contrast held-out set (neg_held_v2) reaches the scorers through protocol_contexts.eval_probe (metadata
# n_neg_train; the contrast-train rows and the fit's rec["neg"] are the run's). Unset: unchanged.
CTX_DIR = os.environ.get("CTX_DIR")
if CTX_DIR:
    if os.environ.get("EVAL_SPLIT", "explicit") != "explicit":
        raise SystemExit("CTX_DIR needs EVAL_SPLIT=explicit, got %r" % os.environ["EVAL_SPLIT"])
    os.environ["EVAL_SPLIT"] = "explicit"
# GROUP_SPLIT (public families; DAN-157): 1 (default) = build_record's group-aware split + the held-out rule
# (circuit/heldout_rule.py; duplicates only, no document rule), 0 = the per-row split of the 15k run. TuringLLM ignores it.
GROUP_SPLIT = PUBLIC and os.environ.get("GROUP_SPLIT", "1") == "1"
def _data_pass_common():
    """069's common.py, loaded under its own module name (070 has a common.py too)."""
    import importlib.util
    if "data_pass_069_common" in sys.modules:
        return sys.modules["data_pass_069_common"]
    spec = importlib.util.spec_from_file_location("data_pass_069_common", EXP / "069-public-data-pass" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# SAE_PRELOAD (public families only; 082 chain): 0 = lazy dictionaries (config.model.sae_preload False: each (kind,
# layer) SAE loads on first use), 1 = load all at bank construction; unset = 069 configure()'s rule. TuringLLM ignores it.
SAE_PRELOAD = os.environ.get("SAE_PRELOAD")
if PUBLIC:
    _data_pass_common().configure(FAMILY, DATA, STORES, skip_missing_saes=SKIP_MISSING_SAES)
    if SAE_PRELOAD in ("0", "1"):
        from config import config as _config
        _config.model.sae_preload = SAE_PRELOAD == "1"


def interleave(targets, slot):
    """Daniel's shallow/deep pairing (production seed_order="interleaved", discovery_window.py), per shard: sort by
    depth, zip the shallow half with the reversed deep half (L0 with L11, L1 with L10, ...), and rotate by one on odd
    co-tenant slots so processes sharing a GPU sit out of phase (half on a deep target while half are on a shallow
    one) instead of hitting their deep, memory-heavy targets together. Keyed on the GPU slot, not the shard index:
    co-tenants are shards g, g+K, g+2K, ..., which all share one parity when K is even."""
    ts = sorted(targets, key=lambda k: (int(k.split(".")[0]), KIND_RANK.get(k.split(".")[1], 3), k))
    h = (len(ts) + 1) // 2
    a, b = ts[:h], ts[h:][::-1]
    out = [x for i in range(h) for x in ([a[i]] + ([b[i]] if i < len(b) else []))]
    return out[1:] + out[:1] if slot % 2 == 1 and len(out) > 1 else out


def log(s):
    print("[%s %s] %s" % (time.strftime("%H:%M:%S"), TAG, s), flush=True)


def jsonl_keys(path, fields):
    """Done keys from EVERY shard's file of this kind (eval.shard*.jsonl etc.), so a rerun with a different shard
    count or target list (stage 1 -> full) skips work any shard already finished."""
    out = set()
    for p in sorted(path.parent.glob(path.name.replace(TAG, "shard*"))):
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001  (a truncated last line after a kill)
                continue
            if "error" not in r:
                out.add(tuple(r.get(f) for f in fields))
    return out


def shard_jsonl(d, stem):
    """A companion's (random_null, rescore_alpha1, stable_core, 073) output file: d/<stem>.jsonl unsharded (the
    names the 15k-run companions wrote), d/<stem>.shard<i>.jsonl under SHARD=i/k with k > 1."""
    return Path(d) / ("%s.jsonl" % stem if SHARD_K == 1 else "%s.shard%d.jsonl" % (stem, SHARD_I))


def read_jsonl_all(d, stem):
    """Every row of d/<stem>.jsonl and of every shard's d/<stem>.shard*.jsonl (truncated lines skipped), so a resume or
    a report sees the work of any shard count."""
    rows = []
    for p in [Path(d) / ("%s.jsonl" % stem)] + sorted(Path(d).glob("%s.shard*.jsonl" % stem)):
        if not p.exists():
            continue
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return rows


def public_setup():
    """The G dict of amp_eval_pass_v2.setup() for a public family: the family's Inference / SAEBank, the 069 stores
    under STORES, the probe builder over the family's shards, and setup()'s config overrides (eval / probe sizes,
    probe batch 4, position-aware off; protocol_harness.configure sets the rest per fit). M0 carries only the probe
    builder: the public path scores through src (wcm_eval), not through a patched M0."""
    from types import SimpleNamespace
    from circuit.probe_dataset import ProbeDatasetBuilder
    from config import config
    from data.loader import DataLoader
    from model.inference import Inference
    from sae.bank import SAEBank
    from store.latent_stats import latent_stats
    C = _data_pass_common()
    torch.set_float32_matmul_precision("high")                        # TF32, as the 068 gate fits (DAN-36)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(device=device)
    C.load_stores(os.path.abspath(STORES), int(sum(loader._shard_sequence_counts)))
    latent_stats.load(os.path.join(STORES, "latent_stats.pt"))
    inference = Inference(device=device)
    bank = SAEBank(device=device)
    C.check_store_grid(bank)
    pb = ProbeDatasetBuilder(inference, bank, loader)
    kinds = list(bank.kinds)
    avg_acts = torch.zeros((bank.n_layer * len(kinds), bank.d_sae), device=bank.device)
    disc = config.discovery                                             # amp_eval_pass_v2.setup()'s overrides
    disc.probe_sequence_count = 64; disc.eval_sequence_count = 64; disc.eval_batch_size = 16
    disc.probe_batch_size = 4; disc.position_aware = False
    return dict(inference=inference, bank=bank, M0=SimpleNamespace(probe_builder=pb), avg_acts=avg_acts, KINDS=kinds,
                NK=len(kinds), D=int(bank.d_sae), K=int(bank.k), device=device, family=FAMILY)


def check_out_version(out, version):
    """Guard OUT against mixing WCM versions (Runner.fit loads an existing circuit instead of refitting). OUT/WCM_VERSION
    names the version OUT's circuits were fitted under; no marker = v1 (every pre-v2 run). v1 never writes the marker
    (OUT is untouched, as before) but refuses a v2-marked OUT; v2 refuses another version's marker, and an unmarked
    OUT that already holds circuits, then marks OUT as v2."""
    out = Path(out)
    marker = out / "WCM_VERSION"
    have = marker.read_text().strip() if marker.exists() else None
    if have is not None and have != version:
        raise RuntimeError("OUT %s holds WCM %s circuits; this process runs WCM_VERSION=%s" % (out, have, version))
    if version != "v1" and have is None:
        legacy = [d for d in (out / "main" / "circuits", out / "sweep" / "circuits")
                  if d.is_dir() and any(d.rglob("*.pt"))]
        if legacy:
            raise RuntimeError("OUT %s already holds unmarked (v1) circuits in %s; a WCM_VERSION=%s run would reuse "
                               "them. Use a fresh OUT." % (out, legacy[0], version))
        out.mkdir(parents=True, exist_ok=True)
        marker.write_text(version + "\n")
    if WCM_VERSIONS[version]["resample_weight"] > 0:
        # the R settings of OUT's circuits (v2cr): refuse an OUT fitted with other ones
        r_now = {k: WCM_VERSIONS[version][k] for k in ("resample_weight", "resample_pool", "resample_k")}
        r_marker = out / "WCM_R"
        if r_marker.exists():
            r_have = json.loads(r_marker.read_text())
            if r_have != r_now:
                raise RuntimeError("OUT %s holds circuits fitted with R %s; this process runs R %s" % (out, r_have, r_now))
        else:
            r_marker.write_text(json.dumps(r_now) + "\n")
    # the lr overrides (LR_SCHEDULE / LR_PEAK_MULT, recorded as LR_MARK): unmarked = the config's; a marked OUT takes
    # only its own
    s_marker = out / "LR_SCHEDULE"
    s_have = s_marker.read_text().strip() if s_marker.exists() else None
    if s_have != LR_MARK:
        if s_have is not None:
            raise RuntimeError("OUT %s holds circuits fitted with lr %s; this process runs %s"
                               % (out, s_have, LR_MARK or "the config's"))
        if any((out / d / "circuits").is_dir() and any((out / d / "circuits").rglob("*.pt")) for d in ("main", "sweep")):
            raise RuntimeError("OUT %s already holds circuits fitted with the config's lr; LR %s needs a fresh OUT"
                               % (out, LR_MARK))
        out.mkdir(parents=True, exist_ok=True)
        s_marker.write_text(LR_MARK + "\n")
    return version


def ctr_fallback(G, key):
    """amp_eval_pass_v2.neg_ctx_store_fallback for the scorer's provenance label: the target's neg_ctx store row is the
    shared zero-similarity fallback list."""
    from pipeline.component_index import component_idx
    from store.context import neg_ctx
    l, k, i = key.split(".")
    comp = component_idx(int(l), G["KINDS"].index(k), G["NK"])
    ids, vals = neg_ctx.ctx_seq_idx[comp, int(i)], neg_ctx.ctx_seq_val[comp, int(i)]
    return bool((ids > 0).any()) and float(vals.float().abs().sum()) == 0.0


class Runner:
    def __init__(self):
        os.environ.setdefault("TAG", "h100_unused")
        os.environ.setdefault("CTR_SOURCE", "close")
        import protocol_harness as H
        self.H, self.P = H, H.P
        self.family = FAMILY
        if PUBLIC:
            self.V = None                                     # 049's module scorer is TuringLLM-only (module globals)
            self.G = public_setup()
        else:
            import amp_eval_pass_v2 as V
            self.V = V
            self.G = V.setup()
        if CTX_DIR:
            self.ctx_dir = Path(CTX_DIR)
            if not self.ctx_dir.is_dir():
                raise FileNotFoundError("CTX_DIR %s is not a directory" % self.ctx_dir)
            if self.V is not None and self.V.EVAL_SPLIT != "explicit":
                raise RuntimeError("CTX_DIR needs the explicit eval split; amp_eval_pass_v2 was imported with %r"
                                   % self.V.EVAL_SPLIT)
        else:
            self.ctx_dir = OUT / "ctx"; self.ctx_dir.mkdir(parents=True, exist_ok=True)
        self.dev = self.G["device"]
        self.wcm_version = check_out_version(OUT, WCM_VERSION)

    # ---------------------------------------------------------------------------------------------- contexts
    def contexts(self, key):
        """The target's context record (src circuit/protocol_contexts.build_record, which reproduces 059's
        pool_test.build exactly; DAN-78), cached as ctx/<key>.pt = {key: rec}."""
        from circuit.protocol_contexts import build_record
        from pipeline.component_index import component_idx
        path = self.ctx_dir / ("%s.pt" % key)
        if path.exists():
            rec = torch.load(path, weights_only=False)[key]
            if GROUP_SPLIT and rec.get("strong") is not None and "heldout" not in rec:
                raise RuntimeError("%s was built without the group-aware split (GROUP_SPLIT=0 to use it)" % path)
            return rec
        if CTX_DIR:
            raise FileNotFoundError("no record %s in CTX_DIR (records there are never built)" % path)
        l, k, i = self.H.parse(key)
        G = self.G
        loader = G["M0"].probe_builder.loader
        kw = {}
        if GROUP_SPLIT:
            from circuit.protocol_contexts import group_split_for
            if not hasattr(self, "_group_split"):
                self._group_split = group_split_for(DATA, loader)
            kw["group_split"] = self._group_split
        rec = build_record(G["inference"], G["bank"], loader, component_idx(l, G["KINDS"].index(k), G["NK"]), i, **kw)
        torch.save({key: rec}, path)
        return rec

    def train_arm(self, rec):
        """32 + 16 (arm B); thin targets without a mid-band pool fall back to the 48 strongest (arm A)."""
        from circuit.protocol_contexts import training_arm
        return training_arm(rec)

    # ---------------------------------------------------------------------------------------------- fitting
    def method(self, gamma, lam, free_amp, steps=STEPS, rank=None):
        from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
        cfg = self.H.configure(gamma, lam, free_amp, steps)
        if rank is None:                                          # the version's weight (v1 / v2 / v2cr 3e-3, v3 1e-1)
            rank = WCM_RANK[self.wcm_version]
        cfg.discovery.learned_mask.rank_weight = float(rank)      # configure() fixes it at the protocol's 3e-3
        for k_, v_ in WCM_VERSIONS[self.wcm_version].items():    # v1: the config defaults (a no-op); v2: 090's fill
            setattr(cfg.discovery.learned_mask, k_, v_)
        if LR_SCHEDULE:
            cfg.discovery.learned_mask.lr_schedule = LR_SCHEDULE
        lr_mult = LR_PEAK_MULT * WCM_LR_MULT.get(self.wcm_version, 1.0)
        if lr_mult != 1.0:
            if not _BASE_LR:
                _BASE_LR.append(float(cfg.discovery.learned_mask.lr))
            cfg.discovery.learned_mask.lr = _BASE_LR[0] * lr_mult
        if RANK_CUT_K:
            cfg.discovery.learned_mask.rank_mode = "cut"
            cfg.discovery.learned_mask.rank_cut_k = RANK_CUT_K
        if self.wcm_version not in WCM_LR_MULT:                  # v4 / v4nr: the version's censor_temp (above)
            cfg.discovery.learned_mask.censor_temp = CENSOR_T
        cfg.discovery.learned_mask.rank_zero_mult = RANK_ZERO_MULT
        cfg.discovery.learned_mask.rival_weight = RIVAL_W
        cfg.discovery.learned_mask.lam_adapt_eps = LAM_ADAPT_EPS
        cfg.discovery.learned_mask.lam_adapt_rate = LAM_ADAPT_RATE
        cfg.discovery.learned_mask.lam_adapt_frac = LAM_ADAPT_FRAC
        cfg.discovery.learned_mask.lam_adapt_up = LAM_ADAPT_UP
        cfg.discovery.learned_mask.lam_adapt_down = LAM_ADAPT_DOWN
        if R_SEED is not None:
            cfg.discovery.learned_mask.resample_seed = R_SEED
        # DEEP_BS (local runs only, 2026-11-04): the deep-target micro-batch (config deep_batch_size, 4: tuned on
        # TuringLLM's 35 sites). GPT-2's 48 sites at 4 spill past the 16 GB card into shared memory (PCIe speed);
        # gradient accumulation keeps the effective batch, so the fit is unchanged up to churn-level noise. Unset =
        # the config's.
        if os.environ.get("DEEP_BS"):
            cfg.discovery.learned_mask.deep_batch_size = int(os.environ["DEEP_BS"])
        cfg.discovery.learned_mask.necessity_weight = NEC_W
        cfg.discovery.learned_mask.necessity_target = NEC_T
        if LR_WARMUP:
            cfg.discovery.learned_mask.warmup_frac = LR_WARMUP
        return _build_mode_method("ablation_gradient", "mask", self.G["inference"], self.G["bank"], self.G["avg_acts"],
                                  self.G["M0"].probe_builder)

    def fit(self, M, rec, key, arm, path, train_fh=None, label=None):
        """Fit (or load) one circuit. A fresh fit writes its training row (fit stats + per-step curve) to
        train_fh before the circuit is saved, so a kill in between refits and repeats the row (merge keeps the last)."""
        if path.exists():
            return torch.load(path, weights_only=False), 0.0
        from circuit.protocol_contexts import eval_reference, training_probe
        from config import config
        KINDS = self.G["KINDS"]
        pdset = training_probe(rec, self.dev, arm)
        M.build_probe_dataset = lambda comp, i, _p=pdset: _p
        M._floor_negatives = lambda probe_data, comp, i, logger: probe_data.neg_tokens
        # v2 (train_means_source="eval_ref"): the scorer's reference rows for this target, sliced as Runner.score's
        # scorer slices them (TuringLLM: 049's EVAL_SPLIT; public: explicit). Set (or cleared) on every fit.
        lm = config.discovery.learned_mask
        _split = self.V.EVAL_SPLIT if self.V is not None else "explicit"
        M._train_means_ref = (eval_reference(rec, self.dev, split=_split)
                              if lm.train_means_source == "eval_ref" else None)
        # v2cr (resample_weight > 0): the R term's donor contexts, the scorer's reference rows of resample_pool
        # (contrast = the 48 contrast-train rows, which 091 donor_tokens asserted equal to rec["neg"][:48] and to the C
        # term's own contrast rows; activating = the 48 strongest-train rows). Set (or cleared) on every fit.
        M._resample_donors = None
        if float(lm.resample_weight) > 0:
            _ref = M._train_means_ref if M._train_means_ref is not None else eval_reference(rec, self.dev, split=_split)
            M._resample_donors = _ref[1] if lm.resample_pool == "contrast" else _ref[0]
        l, k, i = self.H.parse(key)
        M.last_mask_provenance = None
        t0 = time.time()
        c = M.discover(l * len(KINDS) + KINDS.index(k), i)
        t_fit = time.time() - t0
        if train_fh is not None:
            train_fh.write(json.dumps(train_row(M.last_mask_provenance, seed=key, train_arm=arm, t_fit=round(t_fit, 1),
                                                accepted=c is not None, shard=TAG, wcm_version=self.wcm_version,
                                                **(label or {}))) + "\n")
            train_fh.flush()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(c, path)                                    # None = rejected; saved so resume skips it
        return c, t_fit

    # ---------------------------------------------------------------------------------------------- scoring
    def score(self, c, rec, held, alphas=None):
        """The protocol-v1 eval row on held-out `held`. alphas = {(layer, kind): {latent: alpha}} overrides the
        fitted coefficients (rescore_alpha1: every member at 1)."""
        if PUBLIC:
            from circuit.protocol_contexts import eval_probe
            from eval.wcm_eval import circuit_key, score_circuit
            row = score_circuit(self.G["inference"], self.G["bank"], c, eval_probe(rec, held, self.dev),
                                split="explicit", alphas=alphas, ctr_source="close",
                                ctr_fallback=ctr_fallback(self.G, circuit_key(c)))
        else:
            self.H.patch_eval_contexts(self.G, rec, held)
            row = self.V.score_circuit(c, {}, override_alphas=alphas, skip_roles=True)   # empty live pool: free0_rand
            #                                 = empty circuit (not the paper's random baseline, which is DAN-17)
        row["wcm_version"] = self.wcm_version
        return row

    def spec(self, c, rec, arm_label):
        if PUBLIC:
            from circuit.protocol_contexts import eval_probe
            from eval.wcm_specificity import specificity_rows
            rows = specificity_rows(self.G["inference"], self.G["bank"], c, eval_probe(rec, "strong", self.dev),
                                    arm=arm_label, split="explicit")
        else:
            import specificity as S
            if not hasattr(self, "_taps"):
                self._taps = S.make_patchers(self.G)
            self.H.patch_eval_contexts(self.G, rec, "strong")
            rows = S.score(self.G, c, self.V, self._taps[0], self._taps[1], arm_label)
        for r in rows:
            r["wcm_version"] = self.wcm_version
        return rows


TRAIN_STATS = ("loss_initial", "loss_final", "holdout_data_loss", "n_kept", "mean_m_final", "mean_m_kept", "min_m_kept",
               "amp_stats", "steps", "batch_size_used", "n_train_pos", "n_holdout_pos", "n_train_neg", "n_holdout_neg")
RESAMPLE_STATS = ("resample_weight", "resample_pool", "resample_k", "resample_n_donors", "resample_draws",
                  "resample_fill", "resample_fill_calls")
CURVE_STRIDE = int(os.environ.get("CURVE_STRIDE", "1"))


def train_row(prov, **extra):
    """One fit's training record: the engine's fit stats and its per-step curve (loss = sum of the data terms +
    penalty; per-term zero / floor / pos, rank, member count at the keep threshold, mean gate), 4 significant
    figures, every CURVE_STRIDE-th step plus the last."""
    row = dict(extra)
    if prov is None:
        row["no_provenance"] = True                            # the seed never reached the mask engine
        return row
    row.update({k: prov.get(k) for k in TRAIN_STATS})
    if prov.get("resample_weight"):                            # v2cr only (v1 / v2 rows unchanged)
        row.update({k: prov.get(k) for k in RESAMPLE_STATS})
    curve = prov.get("curve") or {}
    n = len(curve.get("loss") or [])
    idx = list(range(0, n, CURVE_STRIDE)) + ([n - 1] if n and (n - 1) % CURVE_STRIDE else [])

    def thin(v):
        return [v[j] if isinstance(v[j], int) else float("%.4g" % v[j]) for j in idx]

    row["curve"] = {k: ({t: thin(s) for t, s in v.items()} if isinstance(v, dict) else thin(v))
                    for k, v in curve.items()}
    row["curve_steps"] = idx if CURVE_STRIDE > 1 else None
    return row


def n_nodes(c):
    return sum(1 for nd in c.nodes.values() if nd.metadata.get("role") != "seed")


def manifest(key, rec, arm):
    """Provenance of every context the target used: corpus sequence ids (and anchor positions) for the training
    activating contexts, the held-out strongest / mid-band sets, and the contrast contexts split as the engine splits
    them (first n - round(n / 4) train, the rest held out)."""
    def ids_at(pool, idx):
        p = rec.get(pool)
        if p is None or p.get("ids") is None:
            return None
        return [int(p["ids"][j]) for j in idx]

    def anchors_at(pool, idx):
        p = rec.get(pool)
        return None if p is None else [int(p["arg"][j]) for j in idx]

    s, m = rec["strong"], rec.get("mid")
    if arm == "B":
        parts = [("strong", rec["D_train"]), ("mid", rec["B_mid"])]
    else:
        parts = [("strong", s["train"])]
    train_ids, train_anchors = [], []
    for pool, idx in parts:
        ids = ids_at(pool, idx)
        train_ids = None if (ids is None or train_ids is None) else train_ids + ids
        train_anchors += anchors_at(pool, idx)
    neg = rec.get("neg_ids")
    n_ntr = None if neg is None else len(neg) - int(round(len(neg) * 0.25))
    return dict(
        seed=key, arm=arm,
        train_ids=train_ids, train_anchors=train_anchors,
        held_strong_ids=ids_at("strong", s["held"]), held_strong_anchors=anchors_at("strong", s["held"]),
        held_mid_ids=ids_at("mid", m["held"]) if m else None, held_mid_anchors=anchors_at("mid", m["held"]) if m else None,
        eval_reference_ids=ids_at("strong", s["train"]),      # the 48 strongest-train contexts: the A ablation value
        contrast_train_ids=None if neg is None else neg[:n_ntr],
        contrast_held_ids=None if neg is None else neg[n_ntr:],
        n_top=rec.get("n_top"), n_mid=rec.get("n_mid"))


def main_mode(R, targets):
    d = OUT / "main"
    ev_path, sp_path, st_path = d / ("eval.%s.jsonl" % TAG), d / ("spec.%s.jsonl" % TAG), d / ("status.%s.jsonl" % TAG)
    cx_path, tr_path = d / ("contexts.%s.jsonl" % TAG), d / ("train.%s.jsonl" % TAG)
    d.mkdir(parents=True, exist_ok=True)
    ev_done, sp_done = jsonl_keys(ev_path, ("seed", "held")), jsonl_keys(sp_path, ("seed",))
    st_done, cx_done = jsonl_keys(st_path, ("seed",)), jsonl_keys(cx_path, ("seed",))
    M = R.method(GAMMA, LAM, True)
    fe, fs, ft, fc = open(ev_path, "a"), open(sp_path, "a"), open(st_path, "a"), open(cx_path, "a")
    fr = open(tr_path, "a")
    for n_, key in enumerate(targets):
        if (key,) in st_done:
            continue
        st = dict(seed=key, shard=TAG, wcm_version=R.wcm_version)
        t = time.time()
        try:
            rec = R.contexts(key); st["t_ctx"] = round(time.time() - t, 1)
            if rec.get("strong") is None:
                st.update(skip="no_contrast" if rec.get("no_contrast") else "too_few_contexts", n_top=rec.get("n_top"))
                ft.write(json.dumps(st) + "\n"); ft.flush()
                continue
            arm, thin = R.train_arm(rec)
            st.update(arm=arm, thin=thin, n_top=rec["n_top"], n_mid=rec["n_mid"])
            if (key,) not in cx_done:
                fc.write(json.dumps(dict(manifest(key, rec, arm), wcm_version=R.wcm_version)) + "\n"); fc.flush()
            c, t_fit = R.fit(M, rec, key, arm, d / "circuits" / ("%s.pt" % key), train_fh=fr)
            st["t_fit"] = round(t_fit, 1)
            if c is None:
                st["skip"] = "rejected"; ft.write(json.dumps(st) + "\n"); ft.flush(); continue
            st["n"] = n_nodes(c)
            t = time.time()
            for held in ("strong", "mid"):
                if rec.get(held) is None or (key, held) in ev_done:
                    continue
                row = R.score(c, rec, held)
                row.update(held=held, arm=arm, thin=thin, shard=TAG)
                fe.write(json.dumps(row) + "\n"); fe.flush()
            st["t_eval"] = round(time.time() - t, 1)
            t = time.time()
            if not SKIP_SPEC and (key,) not in sp_done:
                for r in R.spec(c, rec, arm):
                    fs.write(json.dumps(r) + "\n")
                fs.flush()
            st["t_spec"] = round(time.time() - t, 1)
            st["ok"] = True
        except Exception as e:  # noqa: BLE001
            st["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        ft.write(json.dumps(st) + "\n"); ft.flush()
        log("%4d/%d %-16s %s n=%s ctx %ss fit %ss eval %ss spec %ss" % (
            n_ + 1, len(targets), key, st.get("error") or st.get("skip") or "ok", st.get("n"), st.get("t_ctx"),
            st.get("t_fit"), st.get("t_eval"), st.get("t_spec")))


def sweep_mode(R, targets):
    d = OUT / "sweep"
    d.mkdir(parents=True, exist_ok=True)
    ev_path = d / ("eval.%s.jsonl" % TAG)
    ev_done = jsonl_keys(ev_path, ("seed", "arm"))
    fe, fr = open(ev_path, "a"), open(d / ("train.%s.jsonl" % TAG), "a")
    for name, weighted, lam, steps, gamma, rank in SWEEP:
        M = R.method(gamma, lam, weighted, steps, rank)
        log("arm %s (weighted=%s, lambda %g, %d steps, gamma %g, rank-keep %g)" % (name, weighted, lam, steps, gamma, rank))
        for key in targets:
            if (key, name) in ev_done:
                continue
            try:
                rec = R.contexts(key)
                if rec.get("strong") is None:
                    continue
                arm, thin = R.train_arm(rec)
                main_c = OUT / "main" / "circuits" / ("%s.pt" % key)
                path = d / "circuits" / name / ("%s.pt" % key)
                if (weighted and abs(lam - LAM) < 1e-12 and steps == STEPS and gamma == GAMMA and rank == RANK
                        and main_c.exists() and not path.exists()):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(torch.load(main_c, weights_only=False), path)        # identical config: reuse
                t = time.time()
                c, t_fit = R.fit(M, rec, key, arm, path, train_fh=fr,
                                 label=dict(arm=name, weighted=weighted, lam=lam, steps=steps, gamma=gamma, rank=rank))
                if c is None:
                    row = dict(seed=key, arm=name, skip="rejected")
                else:
                    row = R.score(c, rec, "strong")
                    row.update(held="strong", arm=name, weighted=weighted, lam=lam, steps=steps, gamma=gamma, rank=rank,
                               train_arm=arm, thin=thin,
                               t_fit=round(t_fit, 1), t_total=round(time.time() - t, 1), shard=TAG)
            except Exception as e:  # noqa: BLE001
                row = dict(seed=key, arm=name, error="%s: %s" % (type(e).__name__, str(e)[:300]))
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row["wcm_version"] = R.wcm_version
            fe.write(json.dumps(row) + "\n"); fe.flush()
            log("%-8s %-16s %s n=%s" % (name, key, row.get("error") or row.get("skip") or "ok", row.get("n")))


def main():
    targets = [t for t in TARGETS.read_text().split() if t][SHARD_I::SHARD_K]
    if ORDER == "interleaved":
        targets = interleave(targets, SLOT)
    if LIMIT:
        targets = targets[:LIMIT]
    log("MODE %s | %d targets | order %s (slot %d) | out %s" % (MODE, len(targets), ORDER, SLOT, OUT))
    log("WCM %s | family %s%s | lambda %g | %d steps%s" % (
        WCM_VERSION, FAMILY, " (data %s, stores %s)" % (DATA, STORES) if PUBLIC else "", LAM, STEPS,
        "" if STEPS == 400 else "  ** NOT the protocol's 400 **"))
    R = Runner()
    (main_mode if MODE == "main" else sweep_mode)(R, targets)
    log("DONE")


if __name__ == "__main__":
    main()
