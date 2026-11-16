"""CPU checks for 120's error patches (real GPT-2 small + OAI v5 SAEs, two IOI prompts):
  1. the error-zeroed attribution's full frame (members=None) gives the target's pre-activation in the error-free model,
     computed independently (a forward whose every upstream site returns decode(encode(x))), and differs from the live
     full-model value;
  2. all-members-at-alpha-1 in the error-zeroed Z frame reproduces the error-free full model;
  3. 116's train patch is extracted from refit_err.py's source and installs.

  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=src python -X utf8 -m pytest experiments/120-known-circuits-errors-zeroed/test_120_cpu.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common120 as C  # noqa: E402
import lib117 as L  # noqa: E402

KEY, LAYER, KIND, LAT = "6.attn.100", 6, "attn", 100


@pytest.fixture(scope="module")
def env():
    from config import config
    from model.inference import Inference
    from sae.public_bank import PublicSAEBank
    config.model.family = "gpt2-small"
    dev = torch.device("cpu")
    inf = Inference(dev, family="gpt2-small")
    bank = PublicSAEBank(device=dev, family="gpt2-small", preload=False)
    task = L.load_task("ioi")
    rows = [task["rows"][i] for i in task["split"]["train"][:2]]
    tok = L.pad_batch([r["tokens"] for r in rows], dev)
    anc = torch.tensor([len(r["tokens"]) - 1 for r in rows])
    roles = [L.roles_of(r) for r in rows]
    live = L.attribution(inf, bank, KEY, tok, anc, roles, members=None)[2]
    C.install_struct_error_zero(L)
    return inf, bank, tok, anc, roles, live


def _error_free_pre(inf, bank, tok, anc):
    from eval.ablation_faithfulness import upstream_sites
    from model.hooks import multi_patch
    from sae.dense import sparse_topk_to_dense
    scope = upstream_sites(bank, LAYER, KIND)
    sae = bank.saes[KIND][LAYER]
    cap = {}

    def tr(li, kd, x):
        if (li, kd) == (LAYER, KIND):
            cap["x"] = x
            return x
        if (li, kd) not in scope:
            return x
        ta, ti = bank.encode(x, kd, li)
        return bank.decode(sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype), kd, li)

    class P:
        def __call__(self, model):
            return multi_patch(model, tr)
    inf.disable_compile()
    try:
        with torch.no_grad():
            inf.forward(tok, patcher=P(), return_activations=False, tokenize_final=False)
    finally:
        inf.enable_compile()
    return (cap["x"] @ sae.encoder.weight[LAT] + sae._get_bias_eff()[LAT])[torch.arange(tok.shape[0]), anc].float()


def test_error_free_full_frame(env):
    inf, bank, tok, anc, roles, live = env
    mass, signed, pre, tk = L.attribution(inf, bank, KEY, tok, anc, roles, members=None)
    ref = _error_free_pre(inf, bank, tok, anc)
    assert torch.allclose(pre, ref, atol=1e-3), (pre, ref)
    assert not torch.allclose(pre, live, atol=1e-2), (pre, live)
    assert sum(mass.values()) > 0
    assert C.STATS["struct_calls"] > 0


def test_all_members_equals_error_free_full(env):
    inf, bank, tok, anc, roles, live = env
    from eval.ablation_faithfulness import upstream_sites
    pre = L.attribution(inf, bank, KEY, tok, anc, roles, members=None)[2]
    mem = {s: {i: 1.0 for i in range(int(bank.d_sae))} for s in upstream_sites(bank, LAYER, KIND)}
    pre2 = L.attribution(inf, bank, KEY, tok, anc, roles, members=mem)[2]
    assert torch.allclose(pre2, pre, atol=1e-3)


def test_train_patch_extracts():
    fn, st = C.train_patch()
    import circuit.instrument.learned_mask as LM
    orig = LM.LearnedMaskPatcher.transform
    try:
        fn()
        assert LM.LearnedMaskPatcher.transform is not orig
        assert st["calls"] == 0
    finally:
        LM.LearnedMaskPatcher.transform = orig
