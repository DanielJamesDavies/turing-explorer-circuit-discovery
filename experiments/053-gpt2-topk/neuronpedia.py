"""Neuronpedia as the examples store for the GPT-2 v5 TopK SAEs.

Gemma Scope 2 shipped examples.safetensors; these SAEs don't, but Neuronpedia
hosts top-activating contexts for all eight sets (4 sites x 32k/128k, 12
layers), keyed by the SAME feature indices SAELens uses. Public API, no key:

  https://www.neuronpedia.org/api/feature/gpt2-small/{L}-{src}_{width}-oai/{idx}

Verified shape (6-res_post_32k-oai #100): 45 activating contexts, each with
`tokens` (STRINGS), `values` (per-token activation), `maxValue` and
`maxValueTokenIndex`; plus `frac_nonzero` and an auto-interp `explanations`
entry on the feature itself.

TOKEN MAPPING: Neuronpedia returns display strings (" region", "\\n"), not ids.
GPT-2's BPE vocabulary writes those as "Ġregion" / "Ċ", so the leading space and
newlines are re-encoded before the vocabulary lookup. `check_seed` then verifies
the stored activations against our own loader — the 052 lesson is that a stored
example store has to be reproduced on the model before it is trusted (labels
too: 3 of 8 auto-interp labels were wrong without activation gating, see
concept-circuits work).
"""
import json
import os
import urllib.request

SRC = {"resid-post": "res_post", "resid-mid": "res_mid", "attn-out": "att", "mlp-out": "mlp"}
WIDTH = os.environ.get("GPT2_SAE_WIDTH", "32k")
CACHE = os.environ.get("NP_CACHE", os.path.expanduser("~/neuronpedia_cache"))


def source_id(kind, layer, width=None):
    return "%d-%s_%s-oai" % (layer, SRC[kind], width or WIDTH)


def feature(kind, layer, idx, width=None):
    """the cached Neuronpedia record for one latent."""
    os.makedirs(CACHE, exist_ok=True)
    sid = source_id(kind, layer, width)
    path = os.path.join(CACHE, "gpt2-small_%s_%d.json" % (sid, idx))
    if os.path.exists(path):
        return json.load(open(path))
    url = "https://www.neuronpedia.org/api/feature/gpt2-small/%s/%d" % (sid, idx)
    req = urllib.request.Request(url, headers={"User-Agent": "turing-research"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.load(r)
    json.dump(d, open(path, "w"))
    return d


def to_ids(tok, tokens):
    """Neuronpedia display strings -> GPT-2 ids (space -> 'Ġ', newline -> 'Ċ')."""
    out = []
    for t in tokens:
        v = t.replace(" ", "Ġ").replace("\n", "Ċ").replace("\t", "ĉ")
        i = tok.convert_tokens_to_ids(v)
        if i is None or i == tok.unk_token_id:
            i = tok(t, add_special_tokens=False).input_ids[:1] or [tok.eos_token_id]
            i = i[0]
        out.append(i)
    return out


def contexts(kind, layer, idx, n=None, width=None, dedupe=True):
    """(ids, anchor, stored_peak) per activating context, strongest first.

    DEDUPED BY SEQUENCE by default (052 does the same). Neuronpedia's contexts
    come from OpenWebText, which is full of repeated boilerplate: measured
    2026-09-20, duplicate rates run 3-33% per latent (resid-post 9 #2000: 42
    contexts, 28 distinct, one string 7 times; resid-post 6 #100: "Could not
    subscribe, try again later" 7 times). Without this, the SAME string lands in
    both train and held-out and the held-out scores mean nothing."""
    d = feature(kind, layer, idx, width)
    rows = []
    for a in d.get("activations", []):
        if not a.get("tokens") or a.get("maxValue", 0) <= 0:
            continue
        rows.append((a["tokens"], int(a["maxValueTokenIndex"]), float(a["maxValue"])))
    rows.sort(key=lambda r: -r[2])
    if dedupe:
        seen, keep = set(), []
        for r in rows:
            k = "".join(r[0])
            if k in seen:
                continue
            seen.add(k); keep.append(r)
        rows = keep
    return rows[:n] if n else rows
