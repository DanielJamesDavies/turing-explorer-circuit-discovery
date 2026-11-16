"""116: SAE-error handling switch for every circuit-only run of the production scorers (DAN-214 §1).

The production scorers (049 amp_eval_pass_v2.score_circuit for TuringLLM, src eval/wcm_eval.score_circuit for the
public models) run every circuit-only measurement (Z / A / C, the empty-circuit baselines, pins) through the engine's
eval.ablation_faithfulness.CircuitOnlyPatcher, which keeps the SAE error x - decode(encode(x)) LIVE at every site by
default. The patcher already has an SFC-style error switch (keep_error_sites / error_means, used by experiment 004).
install() wraps its __init__ so that, by MODE["m"]:

  live  unchanged (keep_error_sites None): the production scoring, the paired baseline
  zero  keep_error_sites = {} : the error of every in-scope (upstream) site is set to 0
  mean  keep_error_sites = {} with error_means: every in-scope site's error is replaced by its mean over the scorer's
        own reference contexts for that run's fill, i.e. the same rows the latent fill's means come from:
          A (site_means = the strongest-train means)   -> error mean over the strongest-train rows
          C (site_means = the contrast-train means)    -> error mean over the contrast-train rows
          Z (zero latent fill, no reference)           -> the token-weighted pool of the two
        (mean over every (row, position), as eval.floors.collect_site_error_means / collect_site_means)

Only CircuitOnlyPatcher runs change. Live-stream edits (necessity's a_{M\\C}, sufficiency to induce, the natural
a_pos reads) are runs of the unmodified model and keep the error, as they must; necessity changes only through its
denominator a_pos - a_empty, a_empty being the zero-ablation empty circuit (now with errors ablated).

Error means are registered by the identity of the latent-means dict the scorer passes on, so a site_means dict that
was not produced by the wrapped collectors raises instead of silently using the wrong mean.
"""
MODE = {"m": "live"}
_REG = {}            # id(latent site-means dict) -> {site: error mean [d_model]}
_CUR = {}            # the current target's last pos / neg error means and token counts


def _err_means(inference, bank, tokens, sites):
    from eval.floors import collect_site_error_means
    return collect_site_error_means(inference, bank, tokens, set(sites)) if sites else {}


def _pool():
    p, n = _CUR.get("pos"), _CUR.get("neg")
    if p is None:
        raise RuntimeError("Z-fill error mean requested before the strongest-train error means were collected")
    if n is None:
        return p
    wp, wn = _CUR["n_pos"], _CUR["n_neg"]
    return {s: (p[s] * wp + n[s] * wn) / (wp + wn) if s in n else p[s] for s in p}


def install():
    import eval.ablation_faithfulness as AF
    import eval.floors as FL
    if getattr(AF.CircuitOnlyPatcher, "_err116", False):
        return
    orig_init = AF.CircuitOnlyPatcher.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        m = MODE["m"]
        if m == "live":
            return
        if self.keep_error_sites is not None:
            raise RuntimeError("caller already set keep_error_sites")
        self.keep_error_sites = set()
        if m == "zero":
            self.error_means = None
        elif m == "mean":
            sm = self.site_means
            if sm is None:
                self.error_means = _pool()
            elif len(sm) == 0:
                self.error_means = {}
            else:
                em = _REG.get(id(sm))
                if em is None:
                    raise RuntimeError("no error means registered for these site means")
                self.error_means = em
        else:
            raise ValueError("MODE must be live / zero / mean, got %r" % m)

    AF.CircuitOnlyPatcher.__init__ = init
    AF.CircuitOnlyPatcher._err116 = True

    orig_anchors, orig_means = FL.collect_site_anchors, FL.collect_site_means
    depth = [0]

    def anchors(inference, bank, tokens, sites, *a, **k):
        out = orig_anchors(inference, bank, tokens, sites, *a, **k)
        if depth[0] == 0 and MODE["m"] == "mean":
            em = _err_means(inference, bank, tokens, sites)
            _REG[id(out[0])] = em
            _CUR.clear()
            _CUR.update(pos=em, n_pos=int(tokens.shape[0]) * int(tokens.shape[1]))
        return out

    def means(inference, bank, tokens, sites, *a, **k):
        depth[0] += 1
        try:
            out = orig_means(inference, bank, tokens, sites, *a, **k)
        finally:
            depth[0] -= 1
        if MODE["m"] == "mean":
            em = _err_means(inference, bank, tokens, sites)
            _REG[id(out)] = em
            _CUR.update(neg=em, n_neg=int(tokens.shape[0]) * int(tokens.shape[1]))
        return out

    FL.collect_site_anchors, FL.collect_site_means = anchors, means
    try:                                   # the public scorer imported the names at module level
        import eval.wcm_eval as WE
        WE.collect_site_anchors, WE.collect_site_means = anchors, means
    except Exception:  # noqa: BLE001
        pass
