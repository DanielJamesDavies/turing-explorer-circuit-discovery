"""062 driver FAMILY / LAM / STEPS / SHARD (DAN-40/41 B1-B2).

Fast tests (always): the driver's import-time settings in a clean subprocess (the default TuringLLM path: lambda 1e-3
bitwise, 400 steps, the sweep grid, nothing from store.* imported, the interleaved order unchanged; LAM / STEPS
envs), KIND_RANK on GPT-2's resid_mid, and the companions' shard-file helpers.

End-to-end (opt-in, RUN_062_E2E=1; CPU, minutes): on the local 069 smoke stores (outputs_public/<family>,
data_public/<family>), driver.py MODE=main at STEPS=8 on two targets per public family, then rescore_alpha1,
random_null, stable_core and 073 strict_eval through the same env.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest experiments/062-h100-protocol-v1/test_062_family.py -q
  RUN_062_E2E=1 ... (same command)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
E073 = HERE.parent / "073-strict-eval"
PY = sys.executable
OLD_KIND_RANK = {"attn": 0, "mlp": 1, "resid": 2}         # the 15k run's driver


def _env(**kw):
    env = {k: v for k, v in os.environ.items()
           if k not in ("FAMILY", "DATA", "STORES", "LAM", "STEPS", "SHARD", "OUT", "TARGETS", "MODE", "LIMIT",
                        "SKIP_MISSING_SAES", "EVAL_SPLIT", "SAE_PRELOAD")}
    env.update(PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _probe(code, **env):
    """Import the driver in a fresh interpreter (its settings are read at import) and return the JSON `code` prints."""
    pre = "import sys, json; sys.path.insert(0, %r); import driver as D\n" % str(HERE)
    out = subprocess.run([PY, "-c", pre + code], cwd=str(REPO), env=_env(**env), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_default_is_the_15k_run_path():
    r = _probe("print(json.dumps(dict(family=D.FAMILY, public=D.PUBLIC, lam_exact=D.LAM == 1e-3, steps=D.STEPS, "
               "gamma=D.GAMMA, rank=D.RANK, sweep=[a[0] for a in D.SWEEP], data=D.DATA, stores=D.STORES, "
               "store_imported=any(m.startswith('store') for m in sys.modules))))")
    assert r["family"] == "turingllm" and not r["public"] and r["lam_exact"]
    assert (r["steps"], r["gamma"], r["rank"]) == (400, 0.25, 3e-3)
    assert r["sweep"] == ["W_0.00025", "W_0.0005", "W_0.001", "W_0.002", "W_0.004",
                          "U_1e-05", "U_3e-05", "U_0.0001", "U_0.0003", "U_0.001"]
    assert r["data"] is None and r["stores"] is None and not r["store_imported"]


def test_lam_and_steps_env():
    r = _probe("print(json.dumps(dict(lam=D.LAM, steps=D.STEPS, sweep=[a[0] for a in D.SWEEP])))", LAM="3e-3",
               STEPS="8")
    assert r["lam"] == 3e-3 and r["steps"] == 8
    assert r["sweep"][2] == "W_0.001" and len(r["sweep"]) == 10     # grid names unchanged (no _s suffix)


def test_sae_preload_env_public_only():
    code = ("from config import config\n"
            "print(json.dumps(dict(preload=config.model.sae_preload, family=config.model.family)))")
    for v in ("0", "1"):                                   # TuringLLM: ignored, config untouched
        assert _probe(code, SAE_PRELOAD=v) == dict(preload=True, family="turingllm")
    data = REPO / "data_public" / "pythia-70m"
    if not any(data.glob("shard_*.npy")):
        pytest.skip("no local pythia-70m smoke shards")
    base = dict(FAMILY="pythia-70m", DATA=data, STORES=REPO / "outputs_public" / "pythia-70m")
    assert _probe(code, **base) == dict(preload=True, family="pythia-70m")          # 069 configure()'s rule
    assert _probe(code, SAE_PRELOAD="0", **base)["preload"] is False
    assert _probe(code, SAE_PRELOAD="0", SKIP_MISSING_SAES="1", **base)["preload"] is False
    assert _probe(code, SAE_PRELOAD="1", SKIP_MISSING_SAES="1", **base)["preload"] is True


def test_interleave_unchanged_on_turingllm_and_orders_resid_mid():
    import random
    keys = ["%d.%s.%d" % (l, k, i) for l in range(12) for k in ("attn", "mlp", "resid") for i in (5, 77)]
    random.Random(0).shuffle(keys)
    code = ("keys = %r\n" % keys
            + "old = %r\n" % OLD_KIND_RANK
            + "new = [D.interleave(keys, s) for s in (0, 1)]\n"
            + "D.KIND_RANK = old\n"
            + "ref = [D.interleave(keys, s) for s in (0, 1)]\n"
            + "D.KIND_RANK = {'attn': 0, 'resid_mid': 1, 'mlp': 2, 'resid': 3}\n"
            + "g = sorted(['3.resid.1', '3.mlp.1', '3.resid_mid.1', '3.attn.1'],"
              " key=lambda k: (int(k.split('.')[0]), D.KIND_RANK.get(k.split('.')[1], 3), k))\n"
            + "print(json.dumps(dict(same=new == ref, gpt2=g)))")
    r = _probe(code)
    assert r["same"]
    assert r["gpt2"] == ["3.attn.1", "3.resid_mid.1", "3.mlp.1", "3.resid.1"]


def test_shard_files(tmp_path):
    code = ("from pathlib import Path\n"
            "d = Path(%r)\n" % str(tmp_path)
            + "p = D.shard_jsonl(d, 'random')\n"
            + "p.write_text(json.dumps(dict(seed='a', draw=0)) + '\\n' + '{\"trunc')\n"
            + "(d / 'random.jsonl').write_text(json.dumps(dict(seed='b', draw=0)) + '\\n')\n"
            + "print(json.dumps(dict(name=p.name, seeds=sorted(r['seed'] for r in D.read_jsonl_all(d, 'random')))))")
    r = _probe(code, SHARD="1/3")
    assert r["name"] == "random.shard1.jsonl" and r["seeds"] == ["a", "b"]
    assert _probe("print(json.dumps(D.shard_jsonl('.', 'alpha1').name))") == "alpha1.jsonl"   # unsharded: run's name


# ----------------------------------------------------------------------------------------------- end to end (opt-in)
E2E = {"pythia-70m": ["1.attn.5489", "2.resid.9392"], "gpt2-small": ["1.resid_mid.22", "1.resid.9750"]}


def _rows(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def _run(script, env, log):
    out = subprocess.run([PY, "-X", "utf8", str(script)], cwd=str(REPO), env=env, capture_output=True, text=True)
    Path(log).write_text(out.stdout + "\n--- stderr ---\n" + out.stderr, encoding="utf-8")
    assert out.returncode == 0, "%s failed; see %s\n%s" % (script, log, out.stderr[-3000:])
    return out.stdout


@pytest.mark.skipif(os.environ.get("RUN_062_E2E") != "1", reason="CPU end-to-end (minutes): RUN_062_E2E=1")
@pytest.mark.parametrize("family", sorted(E2E))
def test_public_end_to_end(family, tmp_path):
    stores, data = REPO / "outputs_public" / family, REPO / "data_public" / family
    if not (stores / "top_ctx.pt").exists() or not data.exists():
        pytest.skip("no local 069 smoke stores for %s" % family)
    keys = E2E[family]
    tf = tmp_path / "targets.txt"
    tf.write_text("\n".join(keys) + "\n")
    out = tmp_path / "out"
    base = dict(FAMILY=family, DATA=data, STORES=stores, OUT=out, STEPS=8, SKIP_MISSING_SAES=1)

    # ---- fit -> score -> specificity (driver main mode)
    _run(HERE / "driver.py", _env(MODE="main", TARGETS=tf, ORDER="list", **base), tmp_path / "main.log")
    st = {r["seed"]: r for r in _rows(out / "main" / "status.shard0.jsonl")}
    assert set(st) == set(keys) and not any("error" in r for r in st.values()), st
    ok = [k for k in keys if st[k].get("ok")]
    assert ok, "no target fitted: %s" % st
    ev = _rows(out / "main" / "eval.shard0.jsonl")
    sp = _rows(out / "main" / "spec.shard0.jsonl")
    tr = _rows(out / "main" / "train.shard0.jsonl")
    k_bank = {"pythia-70m": 16, "gpt2-small": 32}[family]
    for k in ok:
        strong = [r for r in ev if r["seed"] == k and r["held"] == "strong"]
        assert len(strong) == 1
        r = strong[0]
        assert r["all_sites_ablated"] and r["n_train"] == 48 and r["n_ho"] == 16 and r["v"] == 3
        assert r["free0_tk"] is not None and r["ctr_source"] == "close" and isinstance(r["ctr_fallback"], bool)
        assert sorted(x["pi"] for x in sp if x["seed"] == k) == ["A", "C", "Z"]
        assert all(x["K"] == k_bank for x in sp if x["seed"] == k)
        assert [x["steps"] for x in tr if x["seed"] == k] == [8]
        mid = [x for x in ev if x["seed"] == k and x["held"] == "mid"]
        assert all(x["n_train"] == 48 for x in mid)               # explicit split: the held-out slice is the mid set

    # ---- companions, same env
    e = _env(**base)
    _run(HERE / "rescore_alpha1.py", dict(e, OUT_A1=str(tmp_path / "a1")), tmp_path / "a1.log")
    a1 = _rows(tmp_path / "a1" / "alpha1.jsonl")
    assert sorted(r["seed"] for r in a1) == sorted(ok) and not any("error" in r for r in a1)
    assert all(r["alpha_median"] == 1.0 for r in a1 if "skip" not in r)

    _run(HERE / "random_null.py", dict(e, OUT_RN=str(tmp_path / "rn"), TARGETS=str(tf), PRESELECTED="1",
                                       N_TARGETS="2", N_DRAWS="1"), tmp_path / "rn.log")
    rn = _rows(tmp_path / "rn" / "random.jsonl")
    assert sorted(r["seed"] for r in rn) == sorted(ok) and not any("error" in r for r in rn), rn
    assert all(r["n_drawn"] > 0 and r["n_drawn"] <= r["n_ref"] for r in rn)

    _run(HERE / "stable_core.py", dict(e, OUT_SC=str(tmp_path / "sc"), SC_TARGETS=str(tf)), tmp_path / "sc.log")
    sc = _rows(tmp_path / "sc" / "stable_core.jsonl")
    assert {(r["seed"], r["stage"]) for r in sc} == {(k, s) for k in ok for s in ("fit2", "core")}, sc

    _run(E073 / "strict_eval.py", dict(e, OUT73=str(tmp_path / "s73"), TARGETS=str(tf), N_DRAWS="1",
                                       RAND_DIR=str(tmp_path / "rn" / "circuits")), tmp_path / "s73.log")
    s73 = _rows(tmp_path / "s73" / "rows.jsonl")
    assert sorted(r["seed"] for r in s73) == sorted(ok) and not any("error" in r for r in s73), s73
    assert all(r["has_rand"] and "rs_R_rand" in r and "ap_C_wcm_r" in r for r in s73)   # random-null variant scored

    # ---- sharded companion: SHARD=1/2 takes the second target, writes its own file
    _run(HERE / "rescore_alpha1.py", dict(e, OUT_A1=str(tmp_path / "a1s"), SHARD="1/2"), tmp_path / "a1s.log")
    a1s = _rows(tmp_path / "a1s" / "alpha1.shard1.jsonl")
    assert [r["seed"] for r in a1s] == sorted(ok)[1::2]
