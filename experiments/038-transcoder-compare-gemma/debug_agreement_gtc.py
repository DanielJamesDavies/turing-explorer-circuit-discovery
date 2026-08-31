"""Localise the 0.7-3.5% feature disagreement between TL's Gemma-2 and
HF's. It is present at layer 0 (upstream of any MLP substitution), so
it is a model-implementation gap, not the surrogate stream. Candidates:
attention (logit softcapping, sliding window), norm eps/(1+w), the
sqrt(d_model) embedding scale.

Compares, per layer, TL hooks against HF tensors at matched points:
  resid_pre   vs HF hidden_states[L]          (stream entering block L)
  ln2 hook    vs HF pre_ffw_ln out / (1+w)    (the transcoder input)
  mlp_out     vs HF post_ffw_ln out           (what their model substitutes)
and at layer 0, embeddings.

  ../../dev-notes/data/venv-ct/bin/python debug_agreement_gtc.py
"""
import torch

MODEL_ID = "unsloth/gemma-2-2b"
TL_NAME = "google/gemma-2-2b"
TEXT = "The Eiffel Tower is in Paris, the capital of France."
LAYERS = [0, 1, 2, 4]


def rel(a, b):
    return float((a - b).abs().max() / max(float(b.abs().max()), 1e-9))


def main():
    import torch as t
    from circuit_tracer.replacement_model import ReplacementModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=t.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok(TEXT, return_tensors="pt")["input_ids"]
    model = ReplacementModel.from_pretrained(
        TL_NAME, "gemma", device=t.device("cpu"), dtype=t.float32,
        hf_model=hf, tokenizer=tok)
    print("TL cfg: softcap attn %s | final %s | window %s | eps %s | n_layers %s"
          % (getattr(model.cfg, "attn_scores_soft_cap", "?"),
             getattr(model.cfg, "output_logits_soft_cap", "?"),
             getattr(model.cfg, "window_size", "?"),
             getattr(model.cfg, "eps", "?"), model.cfg.n_layers))
    print("HF cfg: softcap attn %s | final %s | window %s | eps %s"
          % (hf.config.attn_logit_softcapping, hf.config.final_logit_softcapping,
             hf.config.sliding_window, hf.config.rms_norm_eps))

    hf_cap = {}
    hs = []
    for L in LAYERS:
        blk = hf.model.layers[L]
        hs.append(blk.pre_feedforward_layernorm.register_forward_hook(
            lambda m, i, o, _l=L: hf_cap.__setitem__(("ln2in", _l), i[0].detach())
            or hf_cap.__setitem__(("ln2", _l), o.detach())))
        hs.append(blk.post_feedforward_layernorm.register_forward_hook(
            lambda m, i, o, _l=L: hf_cap.__setitem__(("mlp", _l), o.detach())))
    with torch.no_grad():
        out = hf(ids, output_hidden_states=True)
    for h in hs:
        h.remove()
    hsd = out.hidden_states

    names = set()
    for L in LAYERS:
        names |= {"blocks.%d.hook_resid_pre" % L, "blocks.%d.ln2.hook_normalized" % L,
                  "blocks.%d.hook_mlp_out" % L, "blocks.%d.hook_resid_mid" % L}
    names.add("hook_embed")
    # the unmodified TL model: run_with_hooks on the underlying HookedTransformer
    # still applies their replacement hooks, so compare what they actually run
    _, cache = model.run_with_cache(ids, names_filter=lambda n: n in names)

    print("\nembed: TL hook_embed vs HF hidden_states[0]: rel %.2e"
          % rel(cache["hook_embed"], hsd[0]))
    print("layer | resid_pre  | ln2hook vs HF/(1+w) | resid_mid vs HF ln2 in | mlp_out vs HF post_ffw")
    for L in LAYERS:
        w = hf.model.layers[L].pre_feedforward_layernorm.weight.data
        r1 = rel(cache["blocks.%d.hook_resid_pre" % L], hsd[L])
        r2 = rel(cache["blocks.%d.ln2.hook_normalized" % L],
                 hf_cap[("ln2", L)] / (1 + w))
        r3 = rel(cache["blocks.%d.hook_resid_mid" % L], hf_cap[("ln2in", L)])
        r4 = rel(cache["blocks.%d.hook_mlp_out" % L], hf_cap[("mlp", L)])
        print("  %2d  | %.2e   |      %.2e        |       %.2e          |    %.2e"
              % (L, r1, r2, r3, r4))
    print("\nREADING: resid_pre ok but resid_mid off => ATTENTION differs;"
          " resid_mid ok but ln2hook off => NORM differs; mlp_out off with"
          " ln2 ok => their substitution/decoder path (expected, surrogate).")


if __name__ == "__main__":
    main()
