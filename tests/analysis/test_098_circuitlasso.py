"""Tests for the CircuitLasso baseline (experiments/098-circuitlasso-baseline; DAN-159).

1  recovery: the regression recovers a planted sparse linear dependency (support, coefficients, intercept)
2  solver: FISTA matches sklearn's Lasso (same objective, columns scaled to unit l2 norm, intercept) at fixed lambdas
3  ranking: the captured top-n is the top-n by |coefficient| at the largest path lambda whose support reaches n
4  matched sizes: per-site captures hold exactly the requested counts; unreachable counts are flagged short
5  integration: table2.py reads lasso rows (attr and fitted) into Table 2 and the DAN-77 points
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
E098 = ROOT / "experiments" / "098-circuitlasso-baseline"
E083 = ROOT / "experiments" / "083-v1-comparison"


def _load(path, alias):
    if not path.exists():
        pytest.skip("experiment code not present: %s" % path)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def LC():
    return _load(E098 / "lasso_core.py", "_lasso_core_098")


def sparse_codes(n, p, k, seed=0):
    """Top-K-like codes: each row has k active latents with positive values (COO triplets)."""
    g = torch.Generator().manual_seed(seed)
    cols = torch.stack([torch.randperm(p, generator=g)[:k] for _ in range(n)])            # [n, k]
    vals = torch.rand(n, k, generator=g) * 2 + 0.1
    rows = torch.arange(n)[:, None].expand(n, k)
    return rows.reshape(-1), cols.reshape(-1), vals.reshape(-1)


def dense(rows, cols, vals, n, p):
    X = torch.zeros(n, p, dtype=torch.float64)
    X[rows, cols] = vals.double()
    return X


def planted(n=3000, p=200, k=20, seed=0, noise=0.01):
    rows, cols, vals = sparse_codes(n, p, k, seed)
    X = dense(rows, cols, vals, n, p)
    beta = torch.zeros(p, dtype=torch.float64)
    true = sorted({3, 17, p // 4, (3 * p) // 5, p - 1})
    beta[true] = torch.tensor([1.5, -2.0, 0.8, 1.0, -0.6], dtype=torch.float64)
    g = torch.Generator().manual_seed(seed + 1)
    y = 0.7 + X @ beta + noise * torch.randn(n, generator=g, dtype=torch.float64)
    return rows, cols, vals, y.float(), beta, true, X


# ---------------------------------------------------------------------------------------------- 1 recovery
class TestRecovery:
    def test_planted_support_coefficients_and_intercept(self, LC):
        rows, cols, vals, y, beta, true, _ = planted()
        D = LC.build_design(rows, cols, vals, y)
        assert D.p == 200 and D.n == 3000
        lmax = LC.lam_max(D)
        sup = {}

        def on_step(k, lam, b, info):
            sup[k] = (lam, b.clone())
            return False
        LC.lasso_path(D, LC.lam_grid(lmax, 10, 1e-4), on_step, tol=1e-9, max_iter=20000)
        # at a moderate lambda the support is exactly the planted set
        hits = [k for k, (lam, b) in sup.items() if set((b != 0).nonzero().flatten().tolist()) == set(true)]
        assert hits, "no path lambda recovered exactly the planted support"
        # at the smallest lambda the coefficients and the intercept are close to the truth (original units)
        bo, b0 = LC.original_units(D, sup[max(sup)][1])
        assert torch.allclose(bo.double()[true], beta[true], atol=0.02)
        assert abs(b0 - 0.7) < 0.02
        yhat = D.ybar + D.mv(sup[max(sup)][1]).double()
        assert LC.r2(D.y, yhat) > 0.999

    def test_lam_max_gives_empty_support_and_just_below_does_not(self, LC):
        rows, cols, vals, y, *_ = planted(n=500, p=50, k=8, seed=3)
        D = LC.build_design(rows, cols, vals, y)
        lm = LC.lam_max(D)
        b, _ = LC.fista(D, lm * 1.0001, tol=1e-10)
        assert int((b != 0).sum()) == 0
        b, _ = LC.fista(D, lm * 0.9, tol=1e-10, max_iter=5000)
        assert int((b != 0).sum()) >= 1


# ---------------------------------------------------------------------------------------------- 2 solver
class TestAgainstSklearn:
    @pytest.mark.parametrize("rel", [0.3, 0.05, 0.01])
    def test_fista_matches_sklearn_lasso(self, LC, rel):
        sk = pytest.importorskip("sklearn.linear_model")
        rows, cols, vals, y, _, _, X = planted(n=800, p=60, k=10, seed=5, noise=0.5)
        D = LC.build_design(rows, cols, vals, y, dtype=torch.float64)
        lam = rel * LC.lam_max(D)
        b, info = LC.fista(D, lam, tol=1e-14, max_iter=50000)
        bo, b0 = LC.original_units(D, b)
        s = X.norm(dim=0).numpy()                                          # unit l2 norm columns, as C.1
        m = sk.Lasso(alpha=lam, fit_intercept=True, tol=1e-12, max_iter=200000).fit(X.numpy() / s, y.double().numpy())
        np.testing.assert_allclose(b.numpy(), m.coef_, atol=2e-5)
        np.testing.assert_allclose(bo.numpy(), m.coef_ / s, atol=2e-5)
        assert abs(b0 - m.intercept_) < 1e-4

    def test_fold_design_shares_columns_and_ignores_absent_ones(self, LC):
        rows, cols, vals, y, *_ = planted(n=300, p=40, k=6, seed=7)
        full = torch.unique(cols)
        keep = rows < 150
        D = LC.build_design(rows[keep], cols[keep], vals[keep], y[:150], col_ids=torch.cat([full, torch.tensor([999])]))
        assert D.p == full.numel() + 1 and float(D.s[-1]) == 0.0
        b, _ = LC.fista(D, 0.01 * LC.lam_max(D), tol=1e-10, max_iter=5000)
        assert float(b[-1]) == 0.0                                          # a column with no entry never enters


# ---------------------------------------------------------------------------------------------- 3 ranking
class TestRanking:
    def test_capture_is_top_n_by_abs_coef_at_first_lambda_reaching_n(self, LC):
        rows, cols, vals, y, *_ = planted(n=1500, p=120, k=12, seed=11, noise=0.3)
        D = LC.build_design(rows, cols, vals, y)
        site_of = (D.col // 40).long()
        sizes = {"a": 3, "b": 10, "c": 30}
        cap = LC.PathCapture(sizes=sizes, counts={}, site_of=site_of, s=D.s, rankings=("coef", "std"))
        hist = []

        def on_step(k, lam, b, info):
            cap.step(k, lam, b)
            hist.append((lam, b.clone()))
            return cap.done()
        LC.lasso_path(D, LC.lam_grid(LC.lam_max(D), 10, 1e-4), on_step, tol=1e-9, max_iter=20000)
        for name, n in sizes.items():
            g = cap.got_sizes[name]
            assert not g["short"] and len(g["sel"]["coef"]) == n
            first = next(i for i, (lam, b) in enumerate(hist) if int((b != 0).sum()) >= n)
            lam, b = hist[first]
            assert g["lam"] == lam and g["support"] == int((b != 0).sum())
            coef = (b / D.s).abs()
            want = sorted((b != 0).nonzero().flatten().tolist(), key=lambda j: (-float(coef[j]), j))[:n]
            assert g["sel"]["coef"].tolist() == want
            want_std = sorted((b != 0).nonzero().flatten().tolist(), key=lambda j: (-float(b[j].abs()), j))[:n]
            assert g["sel"]["std"].tolist() == want_std

    def test_top_ties_by_column(self, LC):
        idx = torch.tensor([7, 2, 5, 9])
        score = torch.zeros(10)
        score[[7, 2, 5, 9]] = torch.tensor([1.0, 3.0, 1.0, 2.0])
        assert LC._top(score, idx, 3).tolist() == [2, 9, 5]


# ---------------------------------------------------------------------------------------------- 4 matched sizes
class TestPerSiteCounts:
    def test_per_site_counts_exact_and_short(self, LC):
        rows, cols, vals, y, *_ = planted(n=2000, p=120, k=12, seed=13, noise=0.5)
        D = LC.build_design(rows, cols, vals, y)
        site_of = (D.col // 40).long()                                     # 3 sites of 40 latents
        counts = {0: 5, 1: 2, 2: 39, 3: 4}                                  # site 3 does not exist: short
        cap = LC.PathCapture(sizes={}, counts=counts, site_of=site_of, s=D.s)
        LC.lasso_path(D, LC.lam_grid(LC.lam_max(D), 10, 1e-3), lambda k, lam, b, i: cap.step(k, lam, b) or cap.done(),
                      tol=1e-8, max_iter=20000)
        cap.finish()
        for j in (0, 1):
            g = cap.got_sites[j]
            assert not g["short"] and len(g["sel"]["coef"]) == counts[j]
            assert set(site_of[g["sel"]["coef"]].tolist()) == {j}
        assert cap.got_sites[3]["short"] and len(cap.got_sites[3]["sel"]["coef"]) == 0
        g2 = cap.got_sites[2]
        assert len(g2["sel"]["coef"]) <= 39 and (g2["short"] == (len(g2["sel"]["coef"]) < 39))

    def test_unreached_global_size_is_short_and_takes_the_whole_support(self, LC):
        rows, cols, vals, y, *_ = planted(n=400, p=60, k=6, seed=17)
        D = LC.build_design(rows, cols, vals, y)
        cap = LC.PathCapture(sizes={"huge": 10_000}, counts={}, site_of=torch.zeros(D.p, dtype=torch.long), s=D.s)
        LC.lasso_path(D, LC.lam_grid(LC.lam_max(D), 5, 1e-1), lambda k, lam, b, i: cap.step(k, lam, b) or False)
        cap.finish()
        g = cap.got_sizes["huge"]
        assert g["short"] and len(g["sel"]["coef"]) == g["support"] == int((cap.last["beta"] != 0).sum())


# ---------------------------------------------------------------------------------------------- 5 table2
def _score(seed, **kw):
    r = dict(seed=seed, n=40, free0_tk=1.0, freeM_topk_tk=1.0, freeN_topk_tk=1.0, phi_sup_blind_tk=0.95,
             phi_cf_alpha_blind_tk=1.0, vacuous_tk=False, ctx_dir="ctx")
    r.update(kw)
    return r


class TestTable2Integration:
    def test_table2_reads_lasso_rows(self, tmp_path, monkeypatch):
        sys.path.insert(0, str(E083))
        T2 = _load(E083 / "table2.py", "_table2_098")
        monkeypatch.setattr(T2.C, "OUT", tmp_path)
        monkeypatch.setattr(T2, "REF", "local")
        (tmp_path / "attr").mkdir()
        (tmp_path / "fitted").mkdir()
        attr = [_score("3.mlp.1", method="lasso", ranking="coef", size_arm="W_0.001", n_target=40),
                _score("9.resid.2", method="lasso", ranking="coef", size_arm="W_0.001", n_target=40, freeM_topk_tk=0.2),
                _score("3.mlp.1", method="lasso", ranking="cv", size_arm="natural", n=900)]
        fit = [_score("3.mlp.1", method="fitted_lasso", ranking="coef", variant=v, size_arm="W_0.001") for v in ("a1", "fit")]
        (tmp_path / "attr" / "rows.lasso.shard0.jsonl").write_text("\n".join(json.dumps(r) for r in attr) + "\n")
        (tmp_path / "fitted" / "rows.lasso.shard0.jsonl").write_text("\n".join(json.dumps(r) for r in fit) + "\n")
        ref = [dict(_score(s), arm="W_0.001") for s in ("3.mlp.1", "9.resid.2")]
        rows = T2.table_rows(ref)
        labels = [lab for _, lab, _ in rows]
        lasso = [lab for lab in labels if "Lasso" in lab]
        assert "sparse regression (CircuitLasso, adapted), natural size (support at the cross-validated λ)" in lasso
        assert "sparse regression (CircuitLasso, adapted), top-n by |Lasso coefficient| (the paper's ranking)" in lasso
        assert [lab for lab in lasso if lab.startswith("Lasso-ranked members")] == [
            "Lasso-ranked members (|coefficient|), per-site counts, α = 1",
            "Lasso-ranked members (|coefficient|), per-site counts, fitted coefficients"]
        assert len(lasso) == 4                                    # no empty frames (e.g. the "std" ranking) kept
        t, _ = T2.build_table(rows)
        row = t[t.row.str.contains("top-n by \\|Lasso coefficient\\|")].iloc[0]
        assert (row["pass"], row["N"]) == (1, 2)
        pts = T2.curve_points(ref)
        assert {"lasso/coef", "lasso/cv", "fitted_lasso/coef/a1", "fitted_lasso/coef/fit"} <= set(pts.series)
