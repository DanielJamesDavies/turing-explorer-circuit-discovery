"""Experiment 081: report.py and figure.py end to end on a synthetic out/ folder (CPU), so both run before any GPU
result exists.

  PYTHONPATH=src python -m pytest experiments/081-task-circuits/test_081_report_cpu.py -q

report.py / figure.py share their names with 080's, so the test puts this folder first on sys.path and drops any
cached "report" / "figure" before importing (a combined pytest run of 080 and 081 then gets 081's).
"""
import gzip
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task_common as C  # noqa: E402


def _row(task, arm, lam, n, z, draw=-1, nec=0.95, c=None):
    r = dict(task=task, arm=arm, lam=str(lam), lam_value=lam, n=n, draw=draw, task_free0=z, task_freeM_topk=z,
             task_freeN_topk=z if c is None else c, task_nec=nec, task_acc_free0=0.9 * z, task_free0_a1=0.05,
             obj_free0=z)
    r["task_pass"] = C.passes(r)
    return r


def _attn(layer, heads, bucket="anchor", w=1.0):
    return dict(layer=layer, kind="attn", latent=layer, alpha=1.0, act={bucket: 1.0}, contrib={bucket: w},
                contrib_anchor=w if bucket == "anchor" else 0.0, n_fire=5, n_items=10,
                heads={bucket: {str(h): v for h, v in heads.items()}})


def _synthetic(out, probe):
    rows = []
    for task in ("ioi", "greater_than", "blimp_wh_island"):
        for lam, n, z in ((1e-2, 40, 0.4), (3e-3, 150, 0.85), (1e-3, 600, 1.0)):
            rows.append(_row(task, "W", lam, n, z))
            for d in range(2):
                rows.append(_row(task, "RND", lam, n, 0.1 + 0.05 * d, draw=d, nec=0.2))
            if task.startswith("blimp"):
                rows.append(_row(task, "W_ZA", lam, n, z, c=0.3))
        for lam, n, z in ((1e-3, 500, 0.3), (1e-4, 5000, 0.7), (1e-5, 30000, 0.95)):
            rows.append(_row(task, "U", lam, n, z))
    out.mkdir(parents=True)
    (out / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ioi = [_attn(9, {6: 5.0, 9: 5.0, 2: 1.0}), _attn(10, {0: 4.0, 3: 1.0}), _attn(7, {3: 2.0, 9: 2.0, 0: 0.1}),
           _attn(8, {6: 1.0, 10: 1.0, 2: 0.1}), _attn(5, {5: 1.0, 0: 0.1}, "s2"), _attn(6, {9: 1.0, 1: 1.0}, "s2"),
           _attn(0, {1: 1.0}, "s2"), _attn(3, {0: 1.0, 4: 3.0}, "s2"), _attn(2, {2: 1.0}, "s1_next"),
           _attn(4, {11: 0.5, 3: 1.0}, "s1_next")]
    gt = ([_attn(l, {h: 1.0}) for l, h in C.GT_HEADS["heads"]]
          + [dict(layer=l, kind="mlp", latent=l, alpha=1.0, act={"anchor": 1.0}, contrib={"anchor": 1.0},
                  contrib_anchor=1.0, n_fire=9, n_items=10) for l in (8, 9, 10, 11)]
          + [dict(layer=l, kind="resid", latent=l, alpha=1.0, act={"y1_yy": 1.0}, contrib={"y1_yy": 0.5},
                  contrib_anchor=0.0, n_fire=5, n_items=10) for l in (0, 1, 2)])
    with gzip.open(out / "profiles.jsonl.gz", "wt") as fh:
        for task, pr in (("ioi", ioi), ("greater_than", gt)):
            fh.write(json.dumps(dict(task=task, arm="W", lam="0.003", lam_value=3e-3, members=pr)) + "\n")
    probe.mkdir(parents=True)

    def held(m, f, ok, cf=True):
        return dict(n=80, mean_clean=m, frac_correct=f, cf_gap=m if cf else None, top1=f if cf else None, crit_mean=ok,
                    crit_frac=ok, crit_cf=ok if cf else None, competent=ok)
    comp = {"ioi": dict(family="ioi", expect=True, competent=True, held=held(2.0, 0.9, True), contrast="original"),
            "greater_than": dict(family="greater_than", expect=True, competent=True, held=held(0.7, 0.95, True),
                                 contrast="original"),
            "blimp_wh_island": dict(family="blimp", expect=None, competent=True, held=held(1.0, 0.8, True, False),
                                    contrast="generated"),
            "factual_known": dict(family="factual", expect=None, competent=False, held=held(-0.3, 0.3, False, False),
                                  contrast="generated")}
    (probe / "competence.json").write_text(json.dumps(dict(tasks=comp)))


def test_report_and_figure_run(tmp_path, monkeypatch, capsys):
    out, probe = tmp_path / "out", tmp_path / "probe"
    _synthetic(out, probe)
    monkeypatch.setenv("OUT", str(out))
    monkeypatch.setenv("OUT_PROBE", str(probe))
    monkeypatch.setenv("NO_062", "1")
    monkeypatch.syspath_prepend(str(HERE))
    for mod in ("report", "figure"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    R = importlib.import_module("report")
    assert Path(R.__file__).resolve().parent == HERE
    R.main()
    text = capsys.readouterr().out
    assert "T1 competent tasks get a passing W circuit at some lambda: SUPPORTED" in text
    assert "T3 random circuits with fitted coefficients fail: SUPPORTED" in text
    assert "T4 structure matches the published GPT-2 circuits (IOI, greater-than): SUPPORTED" in text
    assert "not applicable on gpt2-small" in text
    rep = json.loads((out / "report.json").read_text())
    assert rep["claim2"]["ioi"]["pass_W"] == 150 and rep["verdicts"]["T2"][0] == "SUPPORTED"
    s = rep["sensitivity"]["blimp_wh_island"]
    assert s["W"]["full"] and not s["W_ZA"]["full"] and s["W_ZA"]["za"]
    Fg = importlib.import_module("figure")
    assert Path(Fg.__file__).resolve().parent == HERE
    Fg.main()
    assert (out / "figures" / "task_circuits_Z.png").exists() and (out / "figures" / "task_circuits_Z.pdf").exists()
