"""Experiment 081, the GPU side shared by prepare.py, probe.py, calibrate.py and run.py: one engine (model + SAE bank)
per process.

  MODEL=gpt2-small (default)  model.inference.Inference(family="gpt2-small") (HF gpt2 through the site adapter) and
                              sae.public_bank.PublicSAEBank (jbloom/GPT2-Small-OAI-v5-32k-*-SAEs: Top-K k=32, 32,768
                              latents, kinds attn / resid_mid / mlp / resid, 12 layers = 48 sites; layer-normed input)
  MODEL=turingllm             062's driver.Runner bring-up, as 080
Both use 062's protocol_harness.configure for the production mask configuration.

  batch(task)        the task's arrays as tensors: training items first (the whole training pool), then held-out ones
                     (the first HELDOUT_N only, when set: smoke.sh); multi-token answers as teacher-forced (good, bad)
                     row pairs; the C ablation's prompts (the dataset's own patch prompts of every training item, or
                     the generated contrast pool); valid-position masks (prompt tokens)
  means(B)           the A value (mean over the WHOLE training pool) and the C value (mean over the whole contrast
                     pool), per site, over real prompt tokens, in chunks of MEANS_BS prompts; once per task (cached in
                     B), shared by every fit and every scoring of the task
  fit(...)           one WCM fit with the OUTPUT endpoint on the mask engine (run_learned_mask), production
                     configuration (triple floor Z + C + A, gamma 0.25, free coefficients for W / alpha = 1 for U,
                     AdamW, the config's batch / steps / keep threshold / binarisation) and: objective "logit" with
                     the task's contrast (logit_metric "token" / "set" / "seq"); every SAE site maskable; the training
                     stream = steps x batch presentations drawn from the training pool WITHOUT replacement within an
                     epoch (task_common.presentation_order; 400 x 4 = 1,600, so a pool of >= 1,600 never repeats),
                     passed to the engine in order with holdout_frac 0 (the engine cycles its micro-batches through
                     its rows in order); A / C = means(B) (floor_means / pos_floor_means); rank-keep off; logit_norm
                     "gap"; lr from the run config, weight decay rescaled. W_ZA (sensitivity): the C term's weight is 0
                     (Z + A only). RND: support restricted, no penalty, gates open, no binarisation (062
                     random_null.py).
  score(...)         eval.wcm_eval.score_task on the held-out items, coefficients always applied, A / C = means(B)
  probe(...)         the full model's task metric on the clean prompts and (original contrast) their counterfactuals
                     (in chunks of PROBE_CHUNK rows: [rows, vocab] log-probs never exceed one chunk)
  reps / silence     prepare.py's contrast-set inputs (mean-pooled final-layer residual; candidate metrics, chunked)
  active_latents     per site, latents active at a real prompt token of the training items (RND pool; bool masks)
  profile(...)       per member: clean activation by role, attribution of the fitted objective by role (circuit-only Z
                     frame, d metric / d gain at gain 1), and for GPT-2 attention latents the exact per-head
                     decomposition of their pre-activation (head_drive) by role; on the first PROFILE_N (256) held-out
                     items (a fixed random subset: the held-out order is seeded)
"""
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
D062 = HERE.parent / "062-h100-protocol-v1"
D059 = HERE.parent / "059-context-pool"
os.environ.setdefault("OUT", str(HERE / "out"))
os.environ["EVAL_SPLIT"] = "explicit"
os.environ["CTR_SOURCE"] = "close"
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import task_common as C  # noqa: E402

EVAL_BS = int(os.environ.get("EVAL_BS", 16))   # scorer / probe batch (2026-10-27: env knob; numbers unchanged)
PROFILE_BS = 8                      # even: (good, bad) pairs stay together
PROFILE_N = int(os.environ.get("PROFILE_N", 256))  # held-out items per member profile (0 = all; Python-loop bound)
MEANS_BS = 64                       # prompts per collect_site_means call (dense [B, T, 32768] codes per site)
PROBE_CHUNK = 1024                  # rows per log-prob read in probe / prepare (even: pairs stay together)


# ------------------------------------------------------------------------------------------- per-head drive
def head_drive(z: torch.Tensor, w_o: torch.Tensor, b_o: torch.Tensor, w_rows: torch.Tensor, b_eff: torch.Tensor,
               n_heads: int, eps: float):
    """Exact per-head decomposition of attn-out SAE latents' pre-activations (GPT-2 small, hook_attn_out SAEs, which
    read the SUMMED post-W_O attention output, layer-normalised per token).

      z       [B, T, n_heads * d_head]  the input of attn.c_proj (the heads' outputs, concatenated)
      w_o     [n_heads * d_head, d_model]  c_proj.weight (HF Conv1D: out = z @ w_o + b_o)
      w_rows  [n, d_model] the latents' encoder rows; b_eff [n] their effective biases (b_enc - W_enc b_dec)
    The site input is x_n = (a - mean(a)) / (std(a) + eps), a = sum_h o_h + b_o, o_h = z_h @ w_o[h rows]. Centering is
    linear and the scale is shared by all heads at a token, so
      pre_i = sum_h ((o_h - mean(o_h)) / s) . w_i  +  ((b_o - mean(b_o)) / s) . w_i + b_eff_i      exactly.
    Returns (drive [B, T, n_heads, n], const [B, T, n])."""
    B, T, D = z.shape
    dh = D // n_heads
    zh = z.reshape(B, T, n_heads, dh).float()
    wh = w_o.float().reshape(n_heads, dh, -1)                          # [H, dh, d_model]
    o = torch.einsum("bthk,hkd->bthd", zh, wh)                          # [B, T, H, d_model]
    a = o.sum(2) + b_o.float()
    s = (a - a.mean(-1, keepdim=True)).std(-1, keepdim=True) + eps     # [B, T, 1], torch's unbiased std
    oc = o - o.mean(-1, keepdim=True)
    drive = torch.einsum("bthd,nd->bthn", oc / s[..., None], w_rows.float())
    bc = (b_o.float() - b_o.float().mean())
    const = (bc[None, None, :] / s) @ w_rows.float().T + b_eff.float()
    return drive, const


class Engine:
    def __init__(self, model: str = C.MODEL):
        self.model = model
        if model == "turingllm":
            sys.path.insert(0, str(D062))
            import driver  # noqa: E402  (062's)
            self.R = driver.Runner()
            self.H = self.R.H
            G = self.R.G
            self.inf, self.bank, self.dev = G["inference"], G["bank"], G["device"]
        else:
            sys.path.insert(0, str(D059))
            import protocol_harness as H  # noqa: E402  (059's; config only)
            from config import config
            from model.inference import Inference
            from sae.public_bank import PublicSAEBank
            config.model.family = model
            # precision: float32 by default (053's references; 48 dictionaries of 2 x 32768 x 768 floats are ~9.7 GB
            # resident). DTYPE=bfloat16 halves that if a 16 GB card runs short (a precision change: say so in results)
            dtype = os.environ.get("DTYPE") or None
            config.model.dtype = dtype
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.inf = Inference(self.dev, family=model)
            self.bank = PublicSAEBank(device=self.dev, family=model, preload=True, dtype=dtype)
            config.discovery.probe_batch_size = 4
            self.H = H
            self.R = None
        kinds = list(self.bank.kinds)
        self.kinds = kinds
        self.sites = sorted(((l, k) for l in range(self.bank.n_layer) for k in kinds
                             if self.bank.saes[k][l] is not None), key=lambda s: (s[0], kinds.index(s[1])))

    # ------------------------------------------------------------------------------------------- data
    def _t(self, x, dt=torch.long):
        return torch.tensor(x, dtype=dt, device=self.dev)

    def batch(self, task: dict, contrast: dict = None) -> dict:
        """Tensors for one task: training items then held-out items (2 rows per item in seq mode); the C prompts;
        metric specs. contrast: the prepare.py pool (required when task["contrast"] == "generated")."""
        tr, ho = list(task["split"]["train"]), list(task["split"]["held"])
        if C.heldout_n():
            ho = ho[:C.heldout_n()]                      # smoke: a fixed random subset (the held-out order is seeded)
        mode = task.get("answer_mode") or C.answer_mode(task)
        A = C.task_arrays(task, tr + ho, mode)
        rpi = A["rows_per_item"]
        B = dict(name=task["name"], family=task["family"], mode=mode, items=tr + ho, n_items_train=len(tr),
                 n_items_held=len(ho), n_train=len(tr) * rpi, n_held=len(ho) * rpi, rows_per_item=rpi,
                 tokens=self._t(A["tokens"]), anchors=self._t(A["anchors"]), valid=self._t(A["valid"], torch.bool),
                 metric=task["metric"], logit_metric=task["logit_metric"], vac=task["vac"], width=A["width"],
                 contrast=task["contrast"])
        if mode == "seq":
            B["answers"] = self._t(A["answers"])
        else:
            B["good"], B["bad"] = self._t(A["good"]), self._t(A["bad"])
        roles = []
        for i in tr + ho:
            r = task["rows"][i]
            for _ in range(rpi):
                roles.append(list(r["roles"]))
        B["roles"] = roles
        if task["contrast"] == "original":
            if "cf_tokens" not in A:
                raise ValueError("%s: contrast 'original' but some rows have no patch prompt" % task["name"])
            n = len(tr)
            B["c_tokens"], B["c_valid"] = self._t(A["cf_tokens"][:n]), self._t(A["cf_valid"][:n], torch.bool)
            B["cf_all"] = (self._t(A["cf_tokens"]), self._t(A["cf_anchors"]))      # every item's, for the probe
            B["c_source"] = "original"
        else:
            if contrast is None:
                raise ValueError("%s needs its generated contrast pool (prepare.py)" % task["name"])
            pool = [task["rows"][i]["tokens"] for i in contrast["pool"]]
            if set(contrast["pool"]) & set(ho):
                raise ValueError("%s: contrast pool contains held-out items" % task["name"])
            ct, _, cv = C.pad_right(pool)
            B["c_tokens"], B["c_valid"] = self._t(ct), self._t(cv, torch.bool)
            B["c_source"] = "generated"
        return B

    def site_means(self, tokens: torch.Tensor, valid: torch.Tensor) -> dict:
        """{site: [d_sae]} mean dense code over the valid positions of `tokens`, MEANS_BS prompts per forward
        (collect_site_means densifies every site's codes for the whole batch at once), chunk means weighted by
        their valid-position counts: the mean over every valid position."""
        from eval.floors import collect_site_means
        tot, n = None, 0.0
        skip = int(getattr(self.bank, "mean_skip_positions", 0) or 0)     # BOS-prefix banks (eval.floors)
        for s0 in range(0, int(tokens.shape[0]), MEANS_BS):
            v = valid[s0:s0 + MEANS_BS]
            k = float(v[:, skip:].sum())
            m = collect_site_means(self.inf, self.bank, tokens[s0:s0 + MEANS_BS], set(self.sites), valid=v)
            if tot is None:
                tot = {s: x.double() * k for s, x in m.items()}
            else:
                for s, x in m.items():
                    tot[s] += x.double() * k
            n += k
        return {s: (x / n).float() for s, x in tot.items()}

    def means(self, B: dict) -> dict:
        """{"A": the whole training pool's site means, "C": the whole contrast pool's}, computed once per task."""
        if "means" not in B:
            n = B["n_train"]
            B["means"] = dict(A=self.site_means(B["tokens"][:n], B["valid"][:n]),
                              C=self.site_means(B["c_tokens"], B["c_valid"]))
        return B["means"]

    def presentations(self, B: dict, steps: int, batch_size: int) -> list:
        """The training rows of one fit in presentation order: steps x batch_size rows, the items drawn without
        replacement within an epoch (task_common.presentation_order, seeded by the task name; every fit of a task
        sees the same stream), (good, bad) pairs kept together in seq mode."""
        rpi = B["rows_per_item"]
        n_rows = int(steps) * int(batch_size)
        order = C.presentation_order(B["n_items_train"], -(-n_rows // rpi), seed=B["name"])
        return [i * rpi + k for i in order for k in range(rpi)][:n_rows]

    def metrics(self, B: dict) -> dict:
        if B["mode"] == "seq":
            a = B["answers"].cpu()
            return {C.PRIMARY: ("seq_diff", a, None), C.OBJ: ("seq_diff", a, None)}
        return {C.PRIMARY: (B["metric"]["task"], B["good"].cpu(), B["bad"].cpu()),
                C.OBJ: (B["metric"]["obj"], B["good"].cpu(), B["bad"].cpu())}

    # ------------------------------------------------------------------------------------------- fit
    def configure(self, lam: float, weighted: bool, steps: int, lr: float, c_term: bool = True):
        from config import config
        self.H.configure(C.GAMMA, lam, bool(weighted), int(steps))      # 062's production config (sets rank 3e-3)
        lm = config.discovery.learned_mask
        lm.rank_weight = 0.0
        lm.lr = float(lr)
        lm.weight_decay = C.weight_decay(lr)
        if not c_term:
            lm.dual_floor_weight = 0.0                                  # W_ZA: the C (floor) term weighs nothing
        return config

    def engine_kwargs(self, B: dict, lam: float, weighted: bool, steps: int, lr: float, logit_norm: str,
                      support=None, c_term: bool = True) -> dict:
        cfg = self.configure(lam, weighted, steps, lr, c_term)
        lm, disc = cfg.discovery.learned_mask, cfg.discovery
        bs = int(disc.probe_batch_size)
        if (lm.deep_site_threshold is not None and lm.deep_batch_size and len(self.sites) > lm.deep_site_threshold
                and bs % int(lm.deep_batch_size)):
            raise ValueError("the micro-batch %d does not divide the batch %d: the engine's draw would not walk the "
                             "presentation order in steps of one batch" % (lm.deep_batch_size, bs))
        idx = torch.tensor(self.presentations(B, steps, bs), dtype=torch.long, device=B["tokens"].device)
        if B["mode"] == "seq":
            tgt = dict(target_tokens=B["answers"][idx], contrast_tokens=None)
        elif B["logit_metric"] == "set":
            tgt = dict(target_tokens=B["good"][idx], contrast_tokens=B["bad"][idx])
        else:
            tgt = dict(target_tokens=B["good"][idx, 0], contrast_tokens=B["bad"][idx, 0])
        M = self.means(B)
        # holdout_frac 0: every presented row trains (the held-out items are scored separately); the engine's
        # micro-batch s = (step x batch) mod rows walks the steps x batch presentation rows exactly once, in order
        kw = dict(objective="logit", sites=list(self.sites), seed_layer=self.bank.n_layer - 1,
                  seed_kind=self.kinds[-1], seed_latent_idx=0,
                  pos_tokens=B["tokens"][idx], pos_argmax=B["anchors"][idx].cpu(),
                  neg_tokens=B["c_tokens"], pos_valid=B["valid"][idx], neg_valid=B["c_valid"],
                  floor_means=M["C"], pos_floor_means=M["A"],
                  logit_metric=B["logit_metric"], logit_norm=logit_norm, **tgt,
                  mask_floor_source=lm.mask_floor_source, dual_floor_weight=lm.dual_floor_weight,
                  triple_floor_weight=lm.triple_floor_weight, free_amplitude=lm.free_amplitude, amp_l1=lm.amp_l1,
                  offtarget_weight=0.0, rank_weight=0.0, floors_train_only=lm.floors_train_only,
                  signed_amplitude=lm.signed_amplitude, neg_suppress_weight=0.0, margin_topk=None,
                  binarize=lm.binarize, steps=lm.steps, lr=lm.lr, l1_lambda=lm.l1_lambda,
                  keep_threshold=lm.keep_threshold, batch_size=bs, holdout_frac=0.0,
                  theta_init=lm.theta_init, log_every=0, deep_site_threshold=lm.deep_site_threshold,
                  deep_batch_size=lm.deep_batch_size, optimizer=lm.optimizer, weight_decay=lm.weight_decay,
                  code_dtype=lm.code_dtype, lr_schedule=lm.lr_schedule, lr_min_frac=lm.lr_min_frac,
                  warmup_frac=lm.warmup_frac,
                  # WCM_VERSION (v2 = the A / C terms under the scorer's sparsity-preserving fill, straight-through)
                  **C.WCM_VERSIONS[C.WCM_VERSION])
        if support is not None:
            kw.update(support={s: torch.tensor(v, dtype=torch.long) for s, v in support.items()}, l1_lambda=0.0,
                      binarize="none", theta_init=40.0)
        return kw

    def fit(self, B: dict, arm: str, lam: float, lr: float, steps: int, logit_norm: str = C.LOGIT_NORM,
            support=None):
        """(alphas {site: {latent: alpha}}, provenance, t_fit, peak MB)."""
        from circuit.instrument.learned_mask import run_learned_mask
        a = C.ARMS[arm]
        kw = self.engine_kwargs(B, lam, bool(a["weighted"]), steps, lr, logit_norm, support, bool(a["c_term"]))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            scores, prov = run_learned_mask(self.inf, self.bank, **kw)
        finally:
            self.reset()
        t_fit = time.time() - t0
        n_pres = int(kw["pos_tokens"].shape[0]) // B["rows_per_item"]
        prov["present"] = dict(pool=B["n_items_train"], n=n_pres, epochs=round(n_pres / max(1, B["n_items_train"]), 3),
                               c_pool=int(B["c_tokens"].shape[0]))
        prov["wcm_version"] = C.WCM_VERSION
        prov["train_means_source"] = "scorer"     # engine.means(B): the same tensors the scorer uses (= v2's eval_ref)
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None
        if a["weighted"] and prov.get("amp_kept") is not None:
            alphas = C.alphas_from_json(prov["amp_kept"])
        else:
            alphas = defaultdict(dict)
            for f in scores:
                alphas[(int(f.layer), str(f.kind))][int(f.index)] = 1.0
            alphas = dict(alphas)
        return {s: d for s, d in alphas.items() if d}, prov, t_fit, peak

    def reset(self):
        from config import config
        lm = config.discovery.learned_mask
        lm.lr, lm.weight_decay, lm.rank_weight = C.PROD_LR, C.PROD_WD, 3e-3
        lm.dual_floor_weight = C.GAMMA

    # ------------------------------------------------------------------------------------------- score
    def score(self, B: dict, alphas: dict) -> dict:
        from eval import wcm_eval as W
        n = B["n_train"]
        M = self.means(B)
        return W.score_task(self.inf, self.bank, alphas, B["tokens"], B["anchors"], n, B["c_tokens"],
                            self.metrics(B), valid=B["valid"], cf_valid=B["c_valid"], sites=set(self.sites),
                            batch_size=EVAL_BS, vacuous_abs=B["vac"], means_A=M["A"], means_C=M["C"])

    # ------------------------------------------------------------------------------------------- probe
    def item_values(self, tokens, anchors, B, answers=None):
        """Per-item task / obj metric of the full model on `tokens` (one row per item in token / set mode; (good,
        bad) row pairs with `answers` in seq mode)."""
        from eval import wcm_eval as W
        out = {m: dict(values=[], correct=[]) for m in (C.PRIMARY, C.OBJ)}
        top1 = []
        seq = B["mode"] == "seq"
        g_all = None if seq else B["good"].cpu()
        b_all = None if seq else B["bad"].cpu()
        for s0 in range(0, int(tokens.shape[0]), PROBE_CHUNK):     # one chunk of [rows, vocab] log-probs at a time
            s1 = s0 + PROBE_CHUNK
            lp, sq, _ = W.read_rows(self.inf, None, tokens[s0:s1], anchors[s0:s1],
                                    None if answers is None else answers[s0:s1], EVAL_BS)
            for m in (C.PRIMARY, C.OBJ):
                if seq:
                    d = W.pair_diff(sq)
                    out[m]["values"] += d.tolist()
                    out[m]["correct"] += (d > 0).tolist()
                else:
                    g, b = g_all[s0:s1], b_all[s0:s1]
                    kind = B["metric"][m]
                    out[m]["values"] += W.task_metric(lp, kind, g, b).tolist()
                    out[m]["correct"] += W.task_correct(lp, kind, g, b).tolist()
            if not seq:
                top1 += W.task_top1(lp, g_all[s0:s1]).tolist()
            del lp, sq
        out["top1"] = None if seq else top1
        return out

    def probe(self, B: dict, task: dict) -> dict:
        """Per item: the full model's metric on the clean prompt and, for original-contrast tasks, on its
        counterfactual (scored for the clean item's good / bad answers)."""
        ans = B.get("answers")
        clean = self.item_values(B["tokens"], B["anchors"], B, ans)
        cf = None
        if B["contrast"] == "original":
            if B["mode"] == "seq":
                items = B["items"]
                seqs, anchors = [], []
                for i in items:
                    r = task["rows"][i]
                    for a in (r["good"], r["bad"]):
                        seqs.append(list(r["cf_tokens"]) + list(a[:-1]))
                        anchors.append(len(r["cf_tokens"]) - 1)
                t, _, _ = C.pad_right(seqs)
                cf = self.item_values(self._t(t), self._t(anchors), B, ans)
            else:
                cf = self.item_values(B["cf_all"][0], B["cf_all"][1], B)
        return dict(clean=clean, cf=cf)

    # ------------------------------------------------------------------------------------------- contrast inputs
    def reps(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Mean-pooled final-layer residual per prompt (the last layer's "resid" site activation, i.e. the SAE input:
        per-token layer-normalised for GPT-2), over real tokens -> [N, d_model] (CPU)."""
        last, kind = self.bank.n_layer - 1, self.kinds[-1]
        k2i = {k: i for i, k in enumerate(self.kinds)}
        out, cur = [], {}

        def hook(layer_idx, activations):
            if layer_idx == last:
                cur["x"] = activations[k2i[kind]].float()

        self.inf.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                    self.inf.forward(tokens[s0:s0 + EVAL_BS], activations_callback=hook, return_activations=False,
                                     tokenize_final=False)
                    v = valid[s0:s0 + EVAL_BS].to(cur["x"].device).float()[..., None]
                    out.append(((cur["x"] * v).sum(1) / v.sum(1).clamp(min=1)).cpu())
        finally:
            self.inf.enable_compile()
        return torch.cat(out, 0)

    def candidate_metric(self, task: dict, pairs, mode: str) -> dict:
        """{(item, candidate): metric(item's good, item's bad | candidate prompt)} for the given (item, candidate)
        row-id pairs; seq mode teacher-forces each item's answers after the candidate prompt."""
        from eval import wcm_eval as W
        rows = task["rows"]
        out = {}
        if mode != "seq":
            # one forward per distinct candidate, PROBE_CHUNK candidates at a time; each candidate's [vocab] log-probs
            # are read for every item paired with it, then dropped
            by_c = defaultdict(list)
            for i, c in pairs:
                by_c[c].append(i)
            cands = sorted(by_c)
            kind = task["metric"]["task"]
            for c0 in range(0, len(cands), PROBE_CHUNK):
                cc = cands[c0:c0 + PROBE_CHUNK]
                t, a, _ = C.pad_right([rows[c]["tokens"] for c in cc])
                lp, _, _ = W.read_rows(self.inf, None, self._t(t), self._t(a), None, EVAL_BS)
                for j, c in enumerate(cc):
                    items = by_c[c]
                    g = torch.tensor(C.pad_sets([rows[i]["good"] for i in items]))
                    b = torch.tensor(C.pad_sets([rows[i]["bad"] for i in items]))
                    v = W.task_metric(lp[j][None].expand(len(items), -1), kind, g, b).tolist()
                    for i, x in zip(items, v):
                        out[(i, c)] = float(x)
                del lp
            return out
        n_pairs = max(1, PROBE_CHUNK // 2)                          # 2 teacher-forced rows per (item, candidate)
        for p0 in range(0, len(pairs), n_pairs):
            chunk = pairs[p0:p0 + n_pairs]
            seqs, anchors, answers = [], [], []
            for i, c in chunk:
                for ans in (rows[i]["good"], rows[i]["bad"]):
                    seqs.append(list(rows[c]["tokens"]) + list(ans[:-1]))
                    anchors.append(len(rows[c]["tokens"]) - 1)
                    answers.append(list(ans))
            t, _, _ = C.pad_right(seqs)
            _, sq, _ = W.read_rows(self.inf, None, self._t(t), self._t(anchors), torch.tensor(C.pad_sets(answers)),
                                   EVAL_BS)
            d = W.pair_diff(sq).tolist()
            for k, p in enumerate(chunk):
                out[tuple(p)] = d[k]
        return out

    # ------------------------------------------------------------------------------------------- active latents
    def active_latents(self, B: dict) -> dict:
        """Per site, the sorted latents active (> 0) at a real prompt token of any training item (the whole pool; one
        [d_sae] bool mask per site, so memory does not grow with the pool)."""
        n = B["n_train"]
        k2i = {k: i for i, k in enumerate(self.kinds)}
        sites = set(self.sites)
        live = {}
        cur = {}

        def hook(layer_idx, activations):
            for kd in self.kinds:
                s = (layer_idx, kd)
                if s in sites:
                    ta, ti = self.bank.encode(activations[k2i[kd]], kd, layer_idx)
                    m = (ta > 0) & cur["v"].to(ta.device)[..., None]
                    if s not in live:
                        live[s] = torch.zeros(int(self.bank.d_sae), dtype=torch.bool, device=ti.device)
                    live[s][ti[m].long()] = True

        self.inf.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, n, EVAL_BS):
                    s1 = min(n, s0 + EVAL_BS)
                    cur["v"] = B["valid"][s0:s1]
                    self.inf.forward(B["tokens"][s0:s1], activations_callback=hook, return_activations=False,
                                     tokenize_final=False)
        finally:
            self.inf.enable_compile()
        return {s: (live[s].nonzero().flatten().cpu().tolist() if s in live else []) for s in self.sites}

    # ------------------------------------------------------------------------------------------- profile
    def _c_proj(self, layer: int):
        hf = getattr(getattr(self.inf, "model", None), "hf", None)
        try:
            return hf.transformer.h[layer].attn.c_proj
        except AttributeError:
            return None

    def profile(self, B: dict, alphas: dict, rows: str = "held") -> list:
        """Per member: act {bucket: clean activation}, contrib {bucket: attribution of the fitted objective},
        contrib_anchor, n_fire, and for attention members on GPT-2 heads {bucket: {head: summed drive at the
        positions where the member fires}}. bucket = the token's role, "anchor" at the prediction position, "answer"
        at teacher-forced answer positions. Means over items."""
        from eval import wcm_eval as W
        from model.hooks import multi_patch
        from sae.dense import sparse_topk_to_dense
        n_tr, N = B["n_train"], int(B["tokens"].shape[0])
        idx_rows = list(range(n_tr, N)) if rows == "held" else list(range(N))
        if PROFILE_N and rows == "held":                 # the first PROFILE_N held-out items (seeded order)
            idx_rows = idx_rows[:PROFILE_N * B["rows_per_item"]]
        msets = {s: torch.tensor(sorted(d), device=self.dev, dtype=torch.long) for s, d in alphas.items() if d}
        avec = {s: torch.tensor([alphas[s][int(i)] for i in v.tolist()], device=self.dev, dtype=torch.float32)
                for s, v in msets.items()}
        order = [(s, int(i)) for s in sorted(msets) for i in msets[s].tolist()]
        act = defaultdict(lambda: defaultdict(float))
        con = defaultdict(lambda: defaultdict(float))
        heads = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        fire = defaultdict(int)
        k2i = {k: i for i, k in enumerate(self.kinds)}
        bank, D = self.bank, int(self.bank.d_sae)
        sites = set(self.sites)
        seq = B["mode"] == "seq"
        attn_sites = [s for s in msets if s[1] == "attn" and self._c_proj(s[0]) is not None]
        eps = float(getattr(bank, "norm_eps", 1e-5))
        n_heads = int(getattr(getattr(getattr(self.inf, "model", None), "config", None), "n_head", C.N_HEADS))

        class Gain:
            def __init__(self, Bsz, T):
                self.g = {s: torch.ones(Bsz, T, len(v), device=v.device, requires_grad=True) for s, v in msets.items()}

            def __call__(self, model):
                return multi_patch(model, self.transform)

            def transform(self, layer_idx, kd, x):
                s = (layer_idx, kd)
                if s not in sites:
                    return x
                ta, ti = bank.encode(x, kd, layer_idx)
                dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
                error = x - bank.decode(dense, kd, layer_idx)
                patched = torch.zeros_like(dense)
                if s in msets:
                    patched[..., msets[s]] = dense[..., msets[s]] * (self.g[s] * avec[s]).to(dense.dtype)
                return bank.decode(patched, kd, layer_idx) + error

        self.inf.disable_compile()
        try:
            for c0 in range(0, len(idx_rows), PROFILE_BS):
                ri = idx_rows[c0:c0 + PROFILE_BS]
                tk, an, va = B["tokens"][ri], B["anchors"][ri], B["valid"][ri]
                Bsz, T = int(tk.shape[0]), int(tk.shape[1])
                buckets = []
                for j, r in enumerate(ri):
                    rl = list(B["roles"][r]) + ["pad"] * (T - len(B["roles"][r]))
                    a0 = int(an[j])
                    rl[a0] = C.ANCHOR
                    if seq:                                # row = prompt + answer[:-1]: answer tokens at a0+1..a0+L-1
                        n_ans = int((B["answers"][r] >= 0).sum())
                        for t in range(a0 + 1, min(T, a0 + n_ans)):
                            rl[t] = C.ANSWER
                    buckets.append(rl)
                vcpu = va.cpu().clone()
                if seq:                                    # teacher-forced answer positions count as real tokens here
                    for j in range(Bsz):
                        for t in range(T):
                            if buckets[j][t] == C.ANSWER:
                                vcpu[j, t] = True
                cap, zin = {}, {}

                def hook(layer_idx, activations):
                    for kd in self.kinds:
                        s = (layer_idx, kd)
                        if s in msets:
                            ta, ti = bank.encode(activations[k2i[kd]], kd, layer_idx)
                            cap[s] = sparse_topk_to_dense(ta, ti, D, dtype=torch.float32)[..., msets[s]].cpu()

                hs = []
                for s in attn_sites:
                    def pre(_m, args, _l=s[0]):
                        zin[_l] = args[0].detach()
                    hs.append(self._c_proj(s[0]).register_forward_pre_hook(pre))
                try:
                    with torch.no_grad():
                        self.inf.forward(tk, activations_callback=hook, return_activations=False,
                                         tokenize_final=False)
                finally:
                    for h in hs:
                        h.remove()
                hd = {}
                for s in attn_sites:
                    cp = self._c_proj(s[0])
                    sae = bank.saes["attn"][s[0]]
                    w = sae.encoder.weight[msets[s]].detach()
                    be = sae._get_bias_eff()[msets[s]].detach()
                    drv, _ = head_drive(zin[s[0]], cp.weight.detach(), cp.bias.detach(), w, be, n_heads, eps)
                    hd[s] = drv.cpu()                                    # [B, T, H, n]
                p = Gain(Bsz, T)
                out = self.inf.forward(tk, patcher=p, grad_enabled=True, all_logits=True, return_activations=False,
                                       tokenize_final=False)
                lg = out[1] if isinstance(out, (tuple, list)) else out
                if seq:
                    m = W.pair_diff(W.answer_logprob(lg, an, B["answers"][ri])).sum()
                else:
                    rr = torch.arange(Bsz, device=lg.device)
                    lp = torch.log_softmax(lg[rr, an.to(lg.device)].float(), dim=-1)
                    m = W.task_metric(lp, B["metric"]["obj"], B["good"][ri].to(lp.device),
                                      B["bad"][ri].to(lp.device)).sum()
                grads = torch.autograd.grad(m, [p.g[s] for s in sorted(msets)], allow_unused=True)
                gr = {s: (g.detach().cpu() if g is not None else torch.zeros(Bsz, T, len(msets[s])))
                      for s, g in zip(sorted(msets), grads)}
                for s in sorted(msets):
                    a, g = cap[s], gr[s]
                    for j in range(Bsz):
                        for t in range(T):
                            if not bool(vcpu[j, t]):
                                continue
                            bk = buckets[j][t]
                            nz = (a[j, t] != 0).nonzero(as_tuple=True)[0].tolist()
                            for q in nz:
                                act[(s, q)][bk] += float(a[j, t, q])
                                if s in hd:
                                    for h in range(n_heads):
                                        heads[(s, q)][bk][h] += float(hd[s][j, t, h, q])
                            for q in (g[j, t] != 0).nonzero(as_tuple=True)[0].tolist():
                                con[(s, q)][bk] += float(g[j, t, q])
                        for q in (a[j][vcpu[j]] != 0).any(0).nonzero(as_tuple=True)[0].tolist():
                            fire[(s, q)] += 1
                del lg, m, grads, p
        finally:
            self.inf.enable_compile()
        n_items = max(1, len(idx_rows) // B["rows_per_item"])
        prof = []
        for (s, i) in order:
            q = int((msets[s] == i).nonzero()[0])
            row = dict(layer=int(s[0]), kind=str(s[1]), latent=int(i), alpha=float(alphas[s][i]),
                       act={k: v / n_items for k, v in act[(s, q)].items()},
                       contrib={k: v / n_items for k, v in con[(s, q)].items()},
                       contrib_anchor=con[(s, q)].get(C.ANCHOR, 0.0) / n_items, n_fire=fire[(s, q)],
                       n_items=n_items)
            if (s, q) in heads:
                row["heads"] = {bk: {str(h): v / n_items for h, v in hv.items()} for bk, hv in heads[(s, q)].items()}
            prof.append(row)
        return prof
