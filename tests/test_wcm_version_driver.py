"""WCM_VERSION in experiments/062-h100-protocol-v1/driver.py (protocol v2, 2026-10-25): v1 (default) sets the
learned-mask config to its defaults (a no-op), v2 to the sparsity-preserving training fill (straight-through) against
the scorer's means; an unknown version refuses to start; check_out_version keeps one OUT to one version. Subprocesses
keep the env apart (the driver reads it at import)."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
E062 = REPO / "experiments" / "062-h100-protocol-v1"

PROBE = r"""
import json, sys
sys.path.insert(0, %(e062)r)
import driver
from config import config
lm = config.discovery.learned_mask
out = dict(version=driver.WCM_VERSION, table=driver.WCM_VERSIONS,
           defaults=dict(train_fill=lm.train_fill, sparse_grad=lm.sparse_grad, train_means_source=lm.train_means_source,
                         resample_weight=lm.resample_weight, resample_pool=lm.resample_pool, resample_k=lm.resample_k))
print("JSON" + json.dumps(out))
"""

NO_R = dict(resample_weight=0.0, resample_pool="contrast", resample_k=1)

R_GUARD = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, %(e062)r)
import driver
res = {}
out = Path(%(tmp)r) / "o"
res["first"] = driver.check_out_version(out, "v2cr")
res["marker"] = (out / "WCM_VERSION").read_text().strip()
res["r_marker"] = json.loads((out / "WCM_R").read_text())
res["again"] = driver.check_out_version(out, "v2cr")
(out / "WCM_R").write_text(json.dumps(dict(res["r_marker"], resample_k=4)))
try:
    driver.check_out_version(out, "v2cr"); res["other_r"] = "accepted"
except RuntimeError:
    res["other_r"] = "refused"
try:
    driver.check_out_version(out, "v2"); res["v2_on_v2cr"] = "accepted"
except RuntimeError:
    res["v2_on_v2cr"] = "refused"
print("JSON" + json.dumps(res))
"""

GUARD = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, %(e062)r)
import driver
res = {}
def attempt(name, out, version):
    try:
        res[name] = driver.check_out_version(out, version)
    except RuntimeError as e:
        res[name] = "refused"
root = Path(%(tmp)r)
fresh, legacy = root / "fresh", root / "legacy"
(legacy / "main" / "circuits").mkdir(parents=True)
(legacy / "main" / "circuits" / "0.attn.1.pt").write_bytes(b"x")
attempt("v1_fresh", fresh, "v1"); res["v1_marks"] = (fresh / "WCM_VERSION").exists()
attempt("v2_fresh", fresh, "v2"); res["v2_marker"] = (fresh / "WCM_VERSION").read_text().strip()
attempt("v2_again", fresh, "v2")
attempt("v1_on_v2", fresh, "v1")
attempt("v1_legacy", legacy, "v1")
attempt("v2_legacy", legacy, "v2"); res["legacy_marked"] = (legacy / "WCM_VERSION").exists()
print("JSON" + json.dumps(res))
"""


def _run(code, env_extra):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="")
    env.pop("WCM_VERSION", None)
    env.update(env_extra)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO))
    line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
    return p, (json.loads(line[-1][4:]) if line else None)


def test_v1_is_the_default_and_matches_the_config_defaults():
    p, out = _run(PROBE % dict(e062=str(E062)), {})
    assert out is not None, p.stderr[-2000:]
    assert out["version"] == "v1"
    assert out["table"]["v1"] == out["defaults"] == dict(train_fill="dense", sparse_grad="ste",
                                                          train_means_source="fit", **NO_R)
    assert out["table"]["v2"] == dict(train_fill="sparse", sparse_grad="ste", train_means_source="eval_ref", **NO_R)
    # v2cr = v2 + R_C at 095's weight
    assert out["table"]["v2cr"] == dict(out["table"]["v2"], resample_weight=0.25)


def test_v2cr_env_overrides():
    p, out = _run(PROBE % dict(e062=str(E062)), {"WCM_VERSION": "v2cr"})
    assert out is not None and out["version"] == "v2cr", p.stderr[-2000:]
    assert out["table"]["v2cr"]["resample_weight"] == 0.25 and out["table"]["v2"]["resample_weight"] == 0.0
    p, out = _run(PROBE % dict(e062=str(E062)),
                  {"WCM_VERSION": "v2cr", "R_WEIGHT": "0.5", "R_POOL": "activating", "R_K": "4"})
    assert out is not None, p.stderr[-2000:]
    assert out["table"]["v2cr"] == dict(out["table"]["v2"], resample_weight=0.5, resample_pool="activating",
                                        resample_k=4)
    p, out = _run(PROBE % dict(e062=str(E062)), {"WCM_VERSION": "v2", "R_WEIGHT": "0.5"})
    assert out is None and p.returncode != 0 and "v2cr only" in p.stderr
    p, out = _run(PROBE % dict(e062=str(E062)), {"WCM_VERSION": "v2cr", "R_POOL": "random"})
    assert out is None and p.returncode != 0


def test_check_out_version_marks_the_r_settings(tmp_path):
    p, res = _run(R_GUARD % dict(e062=str(E062), tmp=str(tmp_path)), {"WCM_VERSION": "v2cr"})
    assert res is not None, p.stderr[-2000:]
    assert res["first"] == res["again"] == res["marker"] == "v2cr"
    assert res["r_marker"] == dict(resample_weight=0.25, resample_pool="contrast", resample_k=1)
    assert res["other_r"] == "refused" and res["v2_on_v2cr"] == "refused"


def test_v2_and_unknown_versions():
    p, out = _run(PROBE % dict(e062=str(E062)), {"WCM_VERSION": "v2"})
    assert out is not None and out["version"] == "v2", p.stderr[-2000:]
    p, out = _run(PROBE % dict(e062=str(E062)), {"WCM_VERSION": "v9"})
    assert out is None and p.returncode != 0 and "WCM_VERSION" in p.stderr


# ---- v3 (2026-11-01, exp 102's r1.0_rk1e-1): v2cr with R weight 1.0 and rank-keep 1e-1 for every arm
SWEEP_PROBE = r"""
import json, sys
sys.path.insert(0, %(e062)r)
import driver
print("JSON" + json.dumps(dict(version=driver.WCM_VERSION, rank=driver.RANK, wcm_rank=driver.WCM_RANK,
                               sweep=[list(a) for a in driver.SWEEP])))
"""

# Runner.method on CPU without a model: the engine builder is stubbed, so this reads the learned-mask config that
# method() leaves for the fit (protocol_harness.configure + the rank-keep weight + the version's table).
METHOD_PROBE = r"""
import json, sys
from types import SimpleNamespace
sys.path.insert(0, %(e062)r)
import driver
import analysis.circuits.gradient_size_sweep_runner as GS
GS._build_mode_method = lambda *a, **k: "stub"
import protocol_harness as H
from config import config
lm = config.discovery.learned_mask
R = driver.Runner.__new__(driver.Runner)
R.H, R.wcm_version = H, driver.WCM_VERSION
R.G = dict(inference=None, bank=None, avg_acts=None, M0=SimpleNamespace(probe_builder=None))
F = ("rank_weight", "rank_mode", "rank_delta", "resample_weight", "resample_pool", "resample_k", "train_fill",
     "sparse_grad", "train_means_source", "l1_lambda", "steps", "free_amplitude")
snap = lambda: {f: getattr(lm, f, None) for f in F}
res = {}
R.method(driver.GAMMA, driver.LAM, True); res["main"] = snap()
R.method(driver.GAMMA, 1e-4, False); res["unweighted"] = snap()
R.method(driver.GAMMA, driver.LAM, True, driver.STEPS, 3e-3); res["explicit_3e-3"] = snap()
res["sweep"] = {}
for name, w, lam, steps, gamma, rank in driver.SWEEP:
    R.method(gamma, lam, w, steps, rank); res["sweep"][name] = snap()
print("JSON" + json.dumps(res))
"""


def test_v3_table_and_rank():
    p, out = _run(PROBE % dict(e062=str(E062)), {"WCM_VERSION": "v3"})
    assert out is not None and out["version"] == "v3", p.stderr[-2000:]
    assert out["table"]["v3"] == dict(out["table"]["v2"], resample_weight=1.0)
    assert out["table"]["v2cr"] == dict(out["table"]["v2"], resample_weight=0.25)      # v2cr untouched
    for ver, rank in (("v1", 3e-3), ("v2", 3e-3), ("v2cr", 3e-3), ("v3", 1e-1)):
        env = {"WCM_VERSION": ver} if ver != "v1" else {}
        p, s = _run(SWEEP_PROBE % dict(e062=str(E062)), env)
        assert s is not None, p.stderr[-2000:]
        assert s["rank"] == rank and s["wcm_rank"] == dict(v1=3e-3, v2=3e-3, v2cr=3e-3, v3=1e-1, v4=1e-1, v4nr=1e-1)
        # the default grid: 5 WCM + 5 unweighted arms, plain names, every arm at the version's rank-keep weight
        assert [a[0] for a in s["sweep"]] == ["W_0.00025", "W_0.0005", "W_0.001", "W_0.002", "W_0.004",
                                              "U_1e-05", "U_3e-05", "U_0.0001", "U_0.0003", "U_0.001"]
        assert all(a[5] == rank for a in s["sweep"])


def test_v3_refuses_r_overrides():
    for k, v in (("R_WEIGHT", "0.25"), ("R_POOL", "activating"), ("R_K", "4")):
        p, out = _run(PROBE % dict(e062=str(E062)), {"WCM_VERSION": "v3", k: v})
        assert out is None and p.returncode != 0 and "v2cr only" in p.stderr


def test_arm_names_follow_the_versions_rank():
    """W_0.001 means the version's own settings; a _r suffix appears only off the version's rank-keep weight."""
    arms = "W_0.001,W_0.001_r0.1,W_0.001_r0.003,U_0.001,W_1e-4_r1e-1"
    p, s = _run(SWEEP_PROBE % dict(e062=str(E062)), {"WCM_VERSION": "v3", "SWEEP_ARMS": arms})
    assert s is not None, p.stderr[-2000:]
    assert {a[0]: a[5] for a in s["sweep"]} == {"W_0.001": 0.1, "W_0.001_r0.003": 0.003, "U_0.001": 0.1,
                                                "W_0.0001": 0.1}
    p, s = _run(SWEEP_PROBE % dict(e062=str(E062)), {"WCM_VERSION": "v2", "SWEEP_ARMS": arms})
    assert s is not None, p.stderr[-2000:]
    assert {a[0]: a[5] for a in s["sweep"]} == {"W_0.001": 0.003, "W_0.001_r0.1": 0.1, "U_0.001": 0.003,
                                                "W_0.0001_r0.1": 0.1}


def test_method_applies_the_versions_settings_to_every_arm():
    p, res = _run(METHOD_PROBE % dict(e062=str(E062)), {"WCM_VERSION": "v3"})
    assert res is not None, p.stderr[-2000:]
    v3 = dict(rank_weight=0.1, rank_mode="keep", rank_delta=0.0, resample_weight=1.0, resample_pool="contrast",
              resample_k=1, train_fill="sparse", sparse_grad="ste", train_means_source="eval_ref")
    for arm in ["main", "unweighted"] + list(res["sweep"]):
        got = res["sweep"][arm] if arm in res["sweep"] else res[arm]
        assert {k: got[k] for k in v3} == v3, (arm, got)
    assert res["unweighted"]["free_amplitude"] is False and res["main"]["free_amplitude"] is True
    assert res["explicit_3e-3"]["rank_weight"] == 3e-3 and res["explicit_3e-3"]["resample_weight"] == 1.0
    # v2 / v2cr: rank-keep 3e-3 on every arm, R as before
    for ver, w in (("v2", 0.0), ("v2cr", 0.25)):
        p, res = _run(METHOD_PROBE % dict(e062=str(E062)), {"WCM_VERSION": ver})
        assert res is not None, p.stderr[-2000:]
        for got in [res["main"], res["unweighted"]] + list(res["sweep"].values()):
            assert got["rank_weight"] == 3e-3 and got["resample_weight"] == w and got["train_fill"] == "sparse"


def test_check_out_version_v3(tmp_path):
    code = R_GUARD.replace('"v2cr"', '"v3"')
    p, res = _run(code % dict(e062=str(E062), tmp=str(tmp_path)), {"WCM_VERSION": "v3"})
    assert res is not None, p.stderr[-2000:]
    assert res["first"] == res["again"] == res["marker"] == "v3"
    assert res["r_marker"] == dict(resample_weight=1.0, resample_pool="contrast", resample_k=1)
    assert res["other_r"] == "refused" and res["v2_on_v2cr"] == "refused"      # v2 refuses the v3-marked OUT
    cross = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, %(e062)r)
import driver
res = {}
for a, b in (("v3", "v2cr"), ("v2cr", "v3")):
    out = Path(%(tmp)r) / ("x_" + a)
    driver.check_out_version(out, a)
    try:
        driver.check_out_version(out, b); res[a + ">" + b] = "accepted"
    except RuntimeError:
        res[a + ">" + b] = "refused"
print("JSON" + json.dumps(res))
"""
    p, res = _run(cross % dict(e062=str(E062), tmp=str(tmp_path)), {})
    assert res == {"v3>v2cr": "refused", "v2cr>v3": "refused"}, p.stderr[-2000:]


# ---- v4 (2026-11-04, exp 107 latent_pilot): v3 + geometric lr x3.15, warm-up 0.1, censored term; v4nr = v4 - R
V4_FIELDS = ("lr", "lr_schedule", "warmup_frac", "censor_temp")


def test_v4_table():
    p, out = _run(PROBE % dict(e062=str(E062)), {"WCM_VERSION": "v4"})
    assert out is not None and out["version"] == "v4", p.stderr[-2000:]
    assert out["table"]["v4"] == dict(out["table"]["v3"], lr_schedule="geometric", warmup_frac=0.1, censor_temp=0.05)
    assert out["table"]["v4nr"] == dict(out["table"]["v4"], **NO_R)
    assert out["table"]["v3"] == dict(out["table"]["v2"], resample_weight=1.0)          # v3 untouched


def test_v4_method_settings_on_every_arm():
    code = METHOD_PROBE.replace('"sparse_grad", "train_means_source",',
                                '"sparse_grad", "train_means_source", %s,' % ", ".join(repr(f) for f in V4_FIELDS))
    p, base = _run(code % dict(e062=str(E062)), {"WCM_VERSION": "v3"})
    assert base is not None, p.stderr[-2000:]
    lr3 = base["main"]["lr"]
    assert base["main"]["lr_schedule"] == "constant" and base["main"]["censor_temp"] is None
    for ver, w in (("v4", 1.0), ("v4nr", 0.0)):
        p, res = _run(code % dict(e062=str(E062)), {"WCM_VERSION": ver})
        assert res is not None, p.stderr[-2000:]
        for got in [res["main"], res["unweighted"], res["explicit_3e-3"]] + list(res["sweep"].values()):
            assert got["lr_schedule"] == "geometric" and got["warmup_frac"] == 0.1 and got["censor_temp"] == 0.05
            assert abs(got["lr"] - 3.15 * lr3) < 1e-12 and got["resample_weight"] == w      # x3.15 once, not compounded
            assert got["train_fill"] == "sparse"
        assert res["main"]["rank_weight"] == 0.1


def test_v4_refuses_lr_and_censor_overrides(tmp_path):
    for k, v in (("LR_SCHEDULE", "geometric"), ("LR_PEAK_MULT", "3.15"), ("CENSOR_T", "0.05"), ("LR_WARMUP", "0.1")):
        p, out = _run(PROBE % dict(e062=str(E062)), {"WCM_VERSION": "v4", k: v})
        assert out is None and p.returncode != 0 and "fixed under" in p.stderr
    cross = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, %(e062)r)
import driver
res = {}
for a, b in (("v4", "v3"), ("v3", "v4"), ("v4", "v4nr")):
    out = Path(%(tmp)r) / ("x_" + a + b)
    driver.check_out_version(out, a)
    try:
        driver.check_out_version(out, b); res[a + ">" + b] = "accepted"
    except RuntimeError:
        res[a + ">" + b] = "refused"
print("JSON" + json.dumps(res))
"""
    p, res = _run(cross % dict(e062=str(E062), tmp=str(tmp_path)), {})
    assert res == {"v4>v3": "refused", "v3>v4": "refused", "v4>v4nr": "refused"}, p.stderr[-2000:]


def test_check_out_version(tmp_path):
    p, res = _run(GUARD % dict(e062=str(E062), tmp=str(tmp_path)), {})
    assert res is not None, p.stderr[-2000:]
    assert res["v1_fresh"] == "v1" and res["v1_marks"] is False          # v1 never writes into OUT
    assert res["v2_fresh"] == "v2" and res["v2_marker"] == "v2"
    assert res["v2_again"] == "v2"
    assert res["v1_on_v2"] == "refused"
    assert res["v1_legacy"] == "v1"
    assert res["v2_legacy"] == "refused" and res["legacy_marked"] is False
