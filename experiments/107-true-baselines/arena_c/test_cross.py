"""Arena C cross-scoring tests (DAN-196), CPU. In WSL from the repo root:

  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= ./.venv/bin/python -m pytest experiments/107-true-baselines/arena_c/test_cross.py -q -s

  1. SFC's evaluator (score_wcm_sfc.py, in ~/venvs/sfc107) on the WCM "everything" circuit: faithfulness 1
  2. SFC's circuit converted at T_N = 0 under our Z (score_sfc_ours.convert + engine.score): Z = 1 within float error
     (and the every-latent circuit as the exact reference)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_sfc_ours as SO  # noqa: E402
import wcm_marks as WM  # noqa: E402

SFC_PY = Path.home() / "venvs" / "sfc107" / "bin" / "python"


@pytest.mark.skipif(not SFC_PY.exists(), reason="no ~/venvs/sfc107")
def test_sfc_eval_of_everything_is_one():
    p = subprocess.run([str(SFC_PY), str(HERE / "score_wcm_sfc.py"), "--selftest", "--device", "cpu"],
                       capture_output=True, text=True, timeout=1500,
                       env={**{k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
                            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "HF_HUB_OFFLINE": "1"})
    # (their AutoEncoder.from_pretrained t.load()s CUDA-saved files without map_location: the GPU must be visible to
    # unpickle, even though the run itself is on the CPU)
    line = [x for x in p.stdout.splitlines() if x.startswith("JSON")]
    assert line, p.stderr[-3000:]
    r = json.loads(line[0][4:])
    print("RESULT SFC eval of the WCM everything circuit: %s" % {k: round(v["faithfulness"], 6) for k, v in r.items()})
    for v in r.values():
        assert abs(v["faithfulness"] - 1.0) < 1e-3 and abs(v["completeness"]) < 1e-3


def test_sfc_threshold_zero_gives_z_one():
    eng = WM.make_engine("noembed", device="cpu")
    nodes, _ = SO.load_sfc_nodes("simple")
    B = eng.batch(WM.build_task("simple", "sfc"))
    sites = WM.scope_sites("noembed")
    a0, nat = SO.convert(nodes, 0.0, sites)
    allc = {s: {i: 1.0 for i in range(WM.D_SAE)} for s in sites}
    z0 = eng.score(B, a0)["task_free0"]
    za = eng.score(B, allc)["task_free0"]
    n0 = sum(len(d) for d in a0.values())
    # SFC's IE is exactly 0 wherever clean and patch prompts agree (position 0, "The": the IG path has delta 0), so
    # T_N = 0 omits every latent active only there. Adding the latents active at position 0 on the test prompts
    # restores the identity.
    pos0_zero = all(float(act[0].abs().max()) == 0.0 for act, _ in nodes.values())
    import donor_tasks as DN
    ctx_tokens = B["tokens"][B["n_train"]:]
    codes = DN.capture(eng.inf, eng.bank, ctx_tokens, sites)
    aug = {s: dict(a0.get(s, {})) for s in sites}
    for s in sites:
        for i in (codes[s][:, 0] > 0).any(0).nonzero().flatten().tolist():
            aug[s][int(i)] = 1.0
    zp = eng.score(B, aug)["task_free0"]
    print("RESULT SFC T_N = 0 -> %d latents (native: %s): our Z %.6f | IE at position 0 all exactly 0: %s | + latents "
          "active at position 0: %d latents, Z %.6f | every latent: Z %.6f" % (
              n0, nat, z0, pos0_zero, sum(len(d) for d in aug.values()), zp, za))
    assert abs(za - 1.0) < 1e-4
    assert pos0_zero
    assert abs(zp - 1.0) < 1e-3
