"""CPU checks for 117 (real GPT-2 small + OAI v5 SAEs, two IOI prompts):
  1. head_drive_slots is exact: sum_h drive + const + b_eff = the slot's pre-activation (attn-out L3, every slot).
  2. attribution() in the full-model frame leaves the target's pre-activation at its natural value, and in a members-
     only frame with every active latent as a member at alpha 1 it gives the same pre-activation.
  3. attribution() with an empty member set at every site = zero ablation: the gains carry no gradient mass.

  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=src python -X utf8 -m pytest experiments/117-known-circuits-gpt2/test_117_cpu.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lib117 as L  # noqa: E402
from head_drive_lib import head_drive_slots  # noqa: E402


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
    return inf, bank, tok, anc, roles


def test_head_drive_exact(env):
    inf, bank, tok, anc, roles = env
    kinds = list(bank.kinds)
    cap = {}
    h = L.c_proj(inf, 3).register_forward_pre_hook(lambda m, a: cap.__setitem__("z", a[0].detach()))

    def cb(li, acts):
        if li == 3:
            cap["x"] = acts[kinds.index("attn")]
    inf.disable_compile()
    try:
        with torch.no_grad():
            inf.forward(tok, activations_callback=cb, return_activations=False, tokenize_final=False)
    finally:
        h.remove()
        inf.enable_compile()
    sae = bank.saes["attn"][3]
    ta, ti = bank.encode(cap["x"], "attn", 3)
    cp = L.c_proj(inf, 3)
    drv, const = head_drive_slots(cap["z"], cp.weight, cp.bias, sae.encoder.weight, ti, 12, float(bank.norm_eps))
    pre = torch.einsum("btd,btkd->btk", cap["x"].float(), sae.encoder.weight.float()[ti.long()]) \
        + sae._get_bias_eff().float()[ti.long()]
    recon = drv.sum(2) + const + sae._get_bias_eff().float()[ti.long()]
    err = (recon - pre).abs().max() / pre.abs().max()
    assert float(err) < 1e-4, float(err)


def test_full_frame_is_natural(env):
    inf, bank, tok, anc, roles = env
    key = "6.attn.100"
    mass, signed, pre, tk = L.attribution(inf, bank, key, tok, anc, roles, members=None)
    # natural pre via the scorer's convention
    sae = bank.saes["attn"][6]
    kinds = list(bank.kinds)
    cap = {}

    def cb(li, acts):
        if li == 6:
            cap["x"] = acts[kinds.index("attn")]
    with torch.no_grad():
        inf.disable_compile()
        inf.forward(tok, activations_callback=cb, return_activations=False, tokenize_final=False)
        inf.enable_compile()
    nat = (cap["x"] @ sae.encoder.weight[100] + sae._get_bias_eff()[100])[torch.arange(2), anc]
    assert torch.allclose(pre, nat.float(), atol=1e-3), (pre, nat)
    assert sum(mass.values()) > 0
    assert any(k[2] >= 0 for k in mass)                    # head-resolved attention mass exists
    # all-active members at alpha 1 reproduce the full model
    from eval.ablation_faithfulness import upstream_sites
    mem = {s: {i: 1.0 for i in range(int(bank.d_sae))} for s in upstream_sites(bank, 6, "attn")}
    _, _, pre2, _ = L.attribution(inf, bank, key, tok, anc, roles, members=mem)
    assert torch.allclose(pre2, pre, atol=1e-3)


def test_empty_frame(env):
    inf, bank, tok, anc, roles = env
    mass, signed, pre, tk = L.attribution(inf, bank, "4.attn.7", tok, anc, roles, members={})
    assert sum(mass.values()) == 0.0
