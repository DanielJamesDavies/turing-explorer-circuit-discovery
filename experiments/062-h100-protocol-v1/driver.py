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
RANK = 3e-3                                                          # the protocol's rank-keep weight
SWEEP = ([("W_%g" % l, True, l, STEPS, GAMMA, RANK) for l in W_LAMS]
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
    def method(self, gamma, lam, free_amp, steps=STEPS, rank=RANK):
        from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
        cfg = self.H.configure(gamma, lam, free_amp, steps)
        cfg.discovery.learned_mask.rank_weight = float(rank)      # configure() fixes it at the protocol's 3e-3
        return _build_mode_method("ablation_gradient", "mask", self.G["inference"], self.G["bank"], self.G["avg_acts"],
                                  self.G["M0"].probe_builder)

    def fit(self, M, rec, key, arm, path, train_fh=None, label=None):
        """Fit (or load) one circuit. A fresh fit writes its training row (fit stats + per-step curve) to
        train_fh before the circuit is saved, so a kill in between refits and repeats the row (merge keeps the last)."""
        if path.exists():
            return torch.load(path, weights_only=False), 0.0
        from circuit.protocol_contexts import training_probe
        KINDS = self.G["KINDS"]
        pdset = training_probe(rec, self.dev, arm)
        M.build_probe_dataset = lambda comp, i, _p=pdset: _p
        M._floor_negatives = lambda probe_data, comp, i, logger: probe_data.neg_tokens
        l, k, i = self.H.parse(key)
        M.last_mask_provenance = None
        t0 = time.time()
        c = M.discover(l * len(KINDS) + KINDS.index(k), i)
        t_fit = time.time() - t0
        if train_fh is not None:
            train_fh.write(json.dumps(train_row(M.last_mask_provenance, seed=key, train_arm=arm, t_fit=round(t_fit, 1),
                                                accepted=c is not None, shard=TAG, **(label or {}))) + "\n")
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
            return score_circuit(self.G["inference"], self.G["bank"], c, eval_probe(rec, held, self.dev),
                                 split="explicit", alphas=alphas, ctr_source="close",
                                 ctr_fallback=ctr_fallback(self.G, circuit_key(c)))
        self.H.patch_eval_contexts(self.G, rec, held)
        return self.V.score_circuit(c, {}, override_alphas=alphas, skip_roles=True)   # empty live pool: free0_rand =
        #                                       empty circuit (not the paper's random baseline, which is DAN-17)

    def spec(self, c, rec, arm_label):
        if PUBLIC:
            from circuit.protocol_contexts import eval_probe
            from eval.wcm_specificity import specificity_rows
            return specificity_rows(self.G["inference"], self.G["bank"], c, eval_probe(rec, "strong", self.dev),
                                    arm=arm_label, split="explicit")
        import specificity as S
        if not hasattr(self, "_taps"):
            self._taps = S.make_patchers(self.G)
        self.H.patch_eval_contexts(self.G, rec, "strong")
        return S.score(self.G, c, self.V, self._taps[0], self._taps[1], arm_label)


TRAIN_STATS = ("loss_initial", "loss_final", "holdout_data_loss", "n_kept", "mean_m_final", "mean_m_kept", "min_m_kept",
               "amp_stats", "steps", "batch_size_used", "n_train_pos", "n_holdout_pos", "n_train_neg", "n_holdout_neg")
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
        st = dict(seed=key, shard=TAG)
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
                fc.write(json.dumps(manifest(key, rec, arm)) + "\n"); fc.flush()
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
            fe.write(json.dumps(row) + "\n"); fe.flush()
            log("%-8s %-16s %s n=%s" % (name, key, row.get("error") or row.get("skip") or "ok", row.get("n")))


def main():
    targets = [t for t in TARGETS.read_text().split() if t][SHARD_I::SHARD_K]
    if ORDER == "interleaved":
        targets = interleave(targets, SLOT)
    if LIMIT:
        targets = targets[:LIMIT]
    log("MODE %s | %d targets | order %s (slot %d) | out %s" % (MODE, len(targets), ORDER, SLOT, OUT))
    log("family %s%s | lambda %g | %d steps%s" % (
        FAMILY, " (data %s, stores %s)" % (DATA, STORES) if PUBLIC else "", LAM, STEPS,
        "" if STEPS == 400 else "  ** NOT the protocol's 400 **"))
    R = Runner()
    (main_mode if MODE == "main" else sweep_mode)(R, targets)
    log("DONE")


if __name__ == "__main__":
    main()
