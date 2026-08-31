"""One-shot port: 037-gemmascope/ours_gemma.py (SAE harness) ->
ours_gtc.py (transcoder harness). Kept as a file so the transformation
is reviewable and re-runnable."""
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE.parent / "037-gemmascope" / "ours_gemma.py"
DST = HERE / "ours_gtc.py"
s = SRC.read_text(encoding="utf-8")


def rep(old, new, count=1):
    global s
    assert old in s, "MISSING: " + old[:70]
    s = s.replace(old, new, count)


HEADER = '''"""OURS side, on circuit-tracer's HOME TURF: tri-amp mask on Gemma-2-2B
with the GemmaScope JumpReLU TRANSCODERS circuit-tracer ships as its
default "gemma" scan (mwhanna/gemma-scope-transcoders, 16k, one L0 pick
per layer). Both sides read the IDENTICAL safetensors files -- no
conversion -- so the only thing that can differ is method.

CONVENTION, PROBED NOT ASSUMED (probe_gemma_tc.py, gemma_tc_convention.json):
    input   = HF mlp input / (1 + w_pre_ffw)    [TL's unweighted ln2 hook;
              Gemma RMSNorm scales by (1+w), folding w is silently wrong]
    target  = post_feedforward_layernorm(mlp_out) WITH (1+w_post)
              [the MLP's residual contribution]
    gate    = JumpReLU raw, uncentred
    FVU 0.157 at positions >= 1 (BOS dominates a naive FVU: exclude pos 0
    everywhere), measured L0 83.2 vs advertised 88.

TRANSCODER INTERVENTION SEMANTICS (two hooks per layer)
    READ  at pre_feedforward_layernorm output:  c = jumprelu(x_in@W_enc.T+b)
    WRITE at post_feedforward_layernorm output: y <- y + (chat - c) @ W_dec
so an unmodified circuit is exactly a no-op and the transcoder's error
passes through untouched (check_gtc.py certifies this).

Node universe: transcoder features at layers < seed layer.

  PYTHONPATH=. python ours_gtc.py scan
  PYTHONPATH=. python ours_gtc.py run
"""'''
i = s.index('"""'); j = s.index('"""', i + 3) + 3
s = HEADER + s[j:]

rep('''SAE_CACHE = Path(os.environ.get("SAE_CACHE",
                                str(Path.home() / "gemmascope")))
WIDTH = os.environ.get("WIDTH", "16k")
TIER = int(os.environ.get("TIER", 2))       # 0 = sparsest of five''',
    '''TC_DIR = Path(os.environ.get("TC_DIR", str(Path.home() / "gemma_tc")))''')
s = s.replace('"scan_gemma_t%d.pt" % TIER', '"scan_gtc.pt"')
s = s.replace('"ours_gemma_rows_t%d.jsonl" % TIER', '"ours_gtc_rows.jsonl"')
s = s.replace('"ours_gemma_members_t%d.jsonl" % TIER', '"ours_gtc_members.jsonl"')
s = re.sub(r'HERE / \(("[^"]+")\)', r'HERE / \1', s)

TC_BLOCK = '''def block(layer):
    return model.model.layers[layer]


_W_PRE = {}


def w_pre(layer):
    """(1 + w) of this block's pre_feedforward_layernorm: the factor
    between the HF MLP input and TL's unweighted ln2 hook the transcoders
    were trained on."""
    if layer not in _W_PRE:
        w = block(layer).pre_feedforward_layernorm.weight.detach()
        _W_PRE[layer] = (1.0 + w.float()).to(DTYPE)
    return _W_PRE[layer]


def tc(layer):
    if layer not in _TC:
        from safetensors.torch import load_file
        sd = load_file(str(TC_DIR / ("layer_%d.safetensors" % layer)),
                       device=DEV)
        thr = sd["activation_function.threshold"] \\
            if "activation_function.threshold" in sd else sd["threshold"]
        _TC[layer] = {
            "W_enc": sd["W_enc"].to(DTYPE),        # [d_sae, d_model]
            "b_enc": sd["b_enc"].to(DTYPE),
            "W_dec": sd["W_dec"].to(DTYPE),        # [d_sae, d_model]
            "b_dec": sd["b_dec"].to(DTYPE),
            "threshold": thr.to(DTYPE)}
    return _TC[layer]


def pre_acts(layer, x_pre_out):
    """x_pre_out is pre_feedforward_layernorm's OUTPUT (the HF MLP
    input). The transcoder was trained on TL's hook, which sits BEFORE
    the (1+w) weight, so divide it out here. circuit-tracer layout:
    W_enc is [d_sae, d_model], hence the transpose."""
    t = tc(layer)
    x_in = x_pre_out / w_pre(layer)
    return x_in @ t["W_enc"].T + t["b_enc"]


def features(layer, x_pre_out):
    """JumpReLU code on the transcoder input (see pre_acts)."""
    p = pre_acts(layer, x_pre_out)
    return p * (p > tc(layer)["threshold"])


'''
a = s.index("_L0_FOR = {}"); b = s.index("class Runner:")
s = s[:a] + TC_BLOCK + s[b:]

# every capture site hooks the PRE norm (its output is the MLP input)
s = s.replace('(gemma_loader.py)', '(probe_gemma_tc.py)')
s = re.sub(r'post_feedforward_layernorm\s*\.register_forward_hook',
           'pre_feedforward_layernorm.register_forward_hook', s)

RUNNER = '''class Runner:
    """Per-layer transcoder-feature interventions: READ features at the
    pre_feedforward_layernorm output, WRITE the decoded delta at the
    post_feedforward_layernorm output (the tensor the transcoder
    predicts and the residual receives). Captures the seed's
    pre-activation at its own layer's read point."""

    def __init__(self, transforms, seed_layer=None, seed_idx=None):
        self.transforms = transforms          # {layer: fn(feats)->feats}
        self.seed_layer, self.seed_idx = seed_layer, seed_idx
        self.seed_out = None
        self.handles = []
        self._delta = {}

    def _read_hook(self, layer):
        def hook(mod, inp, out):
            fn = self.transforms.get(layer)
            if fn is None:
                return None
            c = features(layer, out)
            chat = fn(c)
            # fp32 mask/amplitude params, bf16 stream: cast the DELTA at
            # the boundary so autograd still reaches the parameters.
            self._delta[layer] = (chat - c).to(out.dtype) @ tc(layer)["W_dec"]
            return None
        return hook

    def _write_hook(self, layer):
        def hook(mod, inp, out):
            d = self._delta.pop(layer, None)
            if d is None:
                return None
            return out + d
        return hook

    def _seed_hook(self):
        def hook(mod, inp, out):
            self.seed_out = pre_acts(self.seed_layer, out)[..., self.seed_idx]
            return None
        return hook

    def __enter__(self):
        for layer in self.transforms:
            blk = block(layer)
            self.handles.append(
                blk.pre_feedforward_layernorm.register_forward_hook(
                    self._read_hook(layer)))
            self.handles.append(
                blk.post_feedforward_layernorm.register_forward_hook(
                    self._write_hook(layer)))
        if self.seed_layer is not None:
            self.handles.append(
                block(self.seed_layer).pre_feedforward_layernorm
                .register_forward_hook(self._seed_hook()))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []
        self._delta = {}


'''
a = s.index("class Runner:"); b = s.index("def forward(tokens, runner, grad=False):")
s = s[:a] + RUNNER + s[b:]

DST.write_text(s, encoding="utf-8", newline="")
import ast; ast.parse(s)
print("wrote", DST.name)
for k in ["TIER", "SAE_CACHE", "_l0_for", "gemma_loader", "% TIER"]:
    print("  leftover %-14s %s" % (k, "YES <-- fix" if k in s else "none"))
print("  pre-hook captures:", s.count("pre_feedforward_layernorm.register_forward_hook"))
print("  post-hook sites:", s.count("post_feedforward_layernorm.register_forward_hook"))
