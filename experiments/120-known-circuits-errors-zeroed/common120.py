"""120 (DAN-215 follow-up): shared paths and the two in-process error patches, both REUSED, not re-implemented.

  train_patch()                116's install_train_error_ablation, taken from refit_err.py's own source by AST (that
                               module cannot be imported: at import it points OUT / CTX_DIR at 116's TuringLLM run and
                               imports 062's driver). It wraps LearnedMaskPatcher.transform so every masked site trains on
                               decode(code) alone (each masked site's SAE error subtracted).
  install_struct_error_zero(L) 117's attribution (lib117.attribution) with every in-scope upstream site's SAE error
                               zeroed: during each attribution call, model.hooks.multi_patch is wrapped so the frame's
                               output decode(edited code) + error at a patched site becomes decode(edited code) (the same
                               subtraction as 116's training patch). Members=None is then the error-free full model
                               (every latent live, every upstream error zeroed). The target site itself is untouched (the
                               attribution stops there), and sites the frame leaves alone (out of scope) are untouched.
Scoring uses 116's errpatch (MODE "zero" / "live") unchanged.
"""
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E116 = ROOT / "experiments" / "116-error-ablated-and-case-checks"
E117 = ROOT / "experiments" / "117-known-circuits-gpt2"
OUT = HERE / "out"
WCM = OUT / "wcm"
STATS = {"train_calls": 0, "struct_calls": 0}

for _p in (str(ROOT / "src"), str(ROOT / "experiments" / "062-h100-protocol-v1"), str(E117), str(E116), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def train_patch():
    """Return 116's install_train_error_ablation (exact source of refit_err.py) bound to a STATS counter."""
    src = (E116 / "refit_err.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "install_train_error_ablation")
    mod = ast.Module(body=[fn], type_ignores=[])
    ns = {"STATS": {"calls": 0}}
    exec(compile(mod, str(E116 / "refit_err.py"), "exec"), ns)
    return ns["install_train_error_ablation"], ns["STATS"]


def install_struct_error_zero(L):
    import model.hooks as MH
    from sae.dense import sparse_topk_to_dense
    if getattr(L, "_err120", False):
        return
    orig_attr, orig_mp = L.attribution, MH.multi_patch
    cur = {}

    def mp(model, transform):
        bank = cur["bank"]

        def t2(li, kd, x):
            out = transform(li, kd, x)
            if out is None or out is x:
                return out
            ta, ti = bank.encode(x, kd, li)
            recon = bank.decode(sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype), kd, li)
            STATS["struct_calls"] += 1
            return out - (x - recon.to(x.dtype))
        return orig_mp(model, t2)

    def attribution(inf, bank, *a, **k):
        cur["bank"] = bank
        MH.multi_patch = mp
        try:
            return orig_attr(inf, bank, *a, **k)
        finally:
            MH.multi_patch = orig_mp

    L.attribution = attribution
    L._err120 = True
