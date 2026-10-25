"""Experiment 080, the GPU side shared by run.py and calibrate.py: fit one circuit for one arm, score it on both
endpoints.

Fits go through 062's driver.Runner.fit (the production path: protocol-v1 training contexts injected into the
ablation-gradient mask method, M.discover, the circuit saved as a .pt), with the mask engine configured per arm before
every fit (the engine reads config.discovery.learned_mask at fit time):
  every arm      062's H.configure: WCM triple floor (Z + C + A), gamma_C = gamma_A = 0.25, ablation values from the
                 training contexts, close contrast, the lambda given, steps
  OUT_* arms     learned_mask.mask_objective = "logit" (log p of the actual next token at each anchor, reproduced
                 against the full model), logit_scope = the arm's scope, logit_norm from the run config, rank_weight 0
                 (rank-keep is about the target latent's rank), lr from the run config with weight decay rescaled so
                 lr x wd stays at production's (common.weight_decay), free_amplitude = the arm's weighted flag
Scoring uses src/eval/wcm_eval.py: score_circuit (latent; the production scorer) and score_output (output), both on the
held-out strongest contexts (eval_probe(rec, "strong"), split "explicit", contrast = the record's close set).
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
D062 = HERE.parent / "062-h100-protocol-v1"
# Before 062's driver is imported: it reads OUT at import (Runner caches contexts under OUT/ctx); the 049 scorer it
# loads reads EVAL_SPLIT / CTR_SOURCE (unused here, set to the protocol's values for consistency).
os.environ.setdefault("OUT", str(HERE / "out"))
os.environ["EVAL_SPLIT"] = "explicit"
os.environ["CTR_SOURCE"] = "close"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(D062))
import torch  # noqa: E402

import common as C  # noqa: E402
import driver  # noqa: E402  (062's)


class Fitter:
    """One engine (model + SAE bank) per process; methods cached per (arm, lambda, lr, steps)."""

    def __init__(self, run_cfg: dict):
        self.cfg = run_cfg
        self.R = driver.Runner()
        self.G = self.R.G
        self.dev = self.G["device"]
        self._methods = {}

    # ------------------------------------------------------------------------------------------- config
    def configure(self, arm: str, lam: float, lr: float, steps: int):
        """Apply the arm's engine config (see the module docstring) and return the cached method for it."""
        from config import config
        a = C.ARMS[arm]
        key = (arm, float(lam), float(lr), int(steps))
        if key not in self._methods:
            self._methods[key] = self.R.method(C.GAMMA, lam, bool(a["weighted"]), int(steps), 0.0)
        self.R.H.configure(C.GAMMA, lam, bool(a["weighted"]), int(steps))
        lm = config.discovery.learned_mask
        lm.rank_weight = 0.0
        lm.mask_objective = "logit"
        lm.logit_scope = str(a["scope"])
        lm.logit_norm = str(self.cfg.get("logit_norm", "gap"))
        lm.lr = float(lr)
        lm.weight_decay = C.weight_decay(lr)
        if self.cfg.get("deep_site_threshold") is not None:        # VRAM guard (gradient accumulation; same regime)
            lm.deep_site_threshold = int(self.cfg["deep_site_threshold"])
            lm.deep_batch_size = int(self.cfg.get("deep_batch_size", 2))
        return self._methods[key]

    @staticmethod
    def reset():
        """Back to the latent endpoint (production values for the fields this module changes)."""
        from config import config
        lm = config.discovery.learned_mask
        lm.mask_objective, lm.logit_scope, lm.logit_norm = "pos", "upstream", "none"
        lm.lr, lm.weight_decay, lm.rank_weight = C.PROD_LR, C.PROD_WD, 3e-3
        lm.deep_site_threshold = 99

    # ------------------------------------------------------------------------------------------- fit
    def fit(self, arm: str, lam: float, lr: float, steps: int, rec: dict, key: str, path: Path, train_fh=None,
            label=None):
        """(circuit or None, t_fit, peak MB). Resumable: an existing file is loaded (t_fit 0)."""
        M = self.configure(arm, lam, lr, steps)
        tr_arm, _ = self.R.train_arm(rec)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            c, t_fit = self.R.fit(M, rec, key, tr_arm, path, train_fh=train_fh, label=label)
        finally:
            self.reset()
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None
        return c, t_fit, peak

    # ------------------------------------------------------------------------------------------- score
    def score(self, c, rec: dict, scope: str = "upstream", latent: bool = True, output: bool = True,
              extra_scopes=()):
        """One merged row: the latent scorer's fields as written by 062 (score_circuit) and the output scorer's
        out_* fields (score_output, the arm's scope; out_n = its member count). extra_scopes: further output scopes,
        their fields prefixed "<scope>_" (e.g. upstream_out_free0 for a whole-model circuit)."""
        from circuit.protocol_contexts import eval_probe
        from eval import wcm_eval as W
        inf, bank = self.G["inference"], self.G["bank"]
        probe = eval_probe(rec, "strong", self.dev)
        row, t = {}, {}
        if latent:
            t0 = time.time()
            alphas = None
            if scope != "upstream":                                  # members the target cannot see are dropped
                ups = W.scope_sites(bank, *C.parse(W.circuit_key(c))[:2], "upstream")
                alphas = {s: d for s, d in W.circuit_alphas(c).items() if s in ups}
            row.update(W.score_circuit(inf, bank, c, probe, split="explicit", alphas=alphas, ctr_source="close"))
            t["t_score_lat"] = round(time.time() - t0, 1)
        if output:
            t0 = time.time()
            o = W.score_output(inf, bank, c, probe, split="explicit", scope=scope)
            o["out_n"] = o.pop("n", None)
            for k in ("seed", "layer", "kind"):
                row.setdefault(k, o.get(k))
                o.pop(k, None)
            if "skip" in o:
                o["out_skip"] = o.pop("skip")
            row.update(o)
            for sc in extra_scopes:
                o2 = W.score_output(inf, bank, c, probe, split="explicit", scope=sc)
                row.update({"%s_%s" % (sc, k): v for k, v in o2.items() if k.startswith("out_")})
            t["t_score_out"] = round(time.time() - t0, 1)
        row.update(t)
        return row


def member_set(c):
    """{(layer, kind, latent)} of a circuit's members (the seed excluded)."""
    if c is None:
        return set()
    return {(int(n.metadata["feature_id"].layer), str(n.metadata["feature_id"].kind), int(n.metadata["feature_id"].index))
            for n in c.nodes.values() if n.metadata.get("role") != "seed"}


def scalars(row: dict) -> dict:
    """The row's JSON-able scalar fields (per_kind and a_pos_blocks kept: small)."""
    keep = {}
    for k, v in row.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            keep[k] = None if isinstance(v, float) and v != v else v
        elif k in ("per_kind", "a_pos_blocks"):
            keep[k] = v
    return keep
