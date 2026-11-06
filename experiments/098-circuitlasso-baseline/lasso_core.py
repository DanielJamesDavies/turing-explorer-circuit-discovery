"""CircuitLasso's solver, adapted to one target latent (exp 098; PAPER_NOTES.md has the paper, README.md the choices).

Pure torch; nothing here loads the model, so the tests run on CPU.

Problem (Yin et al. 2026, Eq. 5 / 6 and App. C.1, restricted to one output column):
    min_b,beta  1/(2n) || y - b - X beta ||^2 + lam ||beta||_1
- X [n samples, p upstream latents] is sparse (Top-K codes), stored as CSR.
- Columns are scaled to unit l2 norm before solving and the coefficients rescaled afterwards (C.1).
- b is an unpenalised intercept. It is handled exactly by implicit centring: the scaled columns are centred inside the
  matrix-vector products, so the design stays sparse.
- FISTA (Beck & Teboulle 2009) with a fixed step 1/L, with L from power iteration, and function-value restart. A
  run stops when the relative objective decrease is below `tol` (C.1: 1e-6).
- Warm-started along a geometric lambda path from lam_max (the smallest lambda with an empty support) downwards (C.1).

The 1/(2n) scaling only re-parameterises lambda (the paper writes 1/2 || . ||^2); lambdas are reported relative to
lam_max, so nothing depends on it.

PathCapture turns the path into member sets:
- global top-n: for each requested size n, the members are the top n by |coefficient| at the largest path lambda whose
  support has at least n latents (the paper ranks by |A| at one lambda);
- per site: the same rule applied within each site, for that site's count.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch


# ---------------------------------------------------------------------------------------------- design
@dataclass
class Design:
    """A sparse regression problem in the scaled, implicitly centred space.

    A   [n, p] CSR, columns scaled to unit l2 norm (uncentred norm, as "unit l2 norm" reads)
    At  [p, n] CSR, the transpose (one fast product each way)
    m   [p]    column means of A (the implicit centring)
    s   [p]    the column norms (beta_orig = beta_scaled / s)
    xm  [p]    column means of the raw X (for the original-units intercept)
    y   [n]    target; ybar its mean; yc = y - ybar
    col [p]    global id of each column (for the caller)
    """
    A: torch.Tensor
    At: torch.Tensor
    m: torch.Tensor
    s: torch.Tensor
    xm: torch.Tensor
    y: torch.Tensor
    ybar: float
    yc: torch.Tensor
    col: torch.Tensor
    n: int
    p: int

    def mv(self, v):
        """A_c v, with A_c = A - 1 m^T (centred scaled design)."""
        return (self.A @ v.unsqueeze(1)).squeeze(1) - torch.dot(self.m, v)

    def rmv(self, r):
        """A_c^T r."""
        return (self.At @ r.unsqueeze(1)).squeeze(1) - self.m * r.sum()


def build_design(rows, cols, vals, y, n=None, col_ids=None, dtype=torch.float32):
    """A Design from COO triplets (rows [nnz] sample index, cols [nnz] global column id, vals [nnz]).

    The columns are the distinct global ids present with a non-zero value (or `col_ids`, a fixed column set, so that a
    cross-validation fold shares the full design's columns; a column with no entry in the fold has norm 0 and a zero
    gradient, so it never enters).
    """
    dev = y.device
    n = int(y.shape[0]) if n is None else int(n)
    keep = vals != 0
    rows, cols, vals = rows[keep].long(), cols[keep].long(), vals[keep].to(dtype)
    if col_ids is None:
        col_ids, inv = torch.unique(cols, return_inverse=True)
    else:
        col_ids = col_ids.to(dev)
        inv = torch.searchsorted(col_ids, cols)
        ok = (inv < col_ids.numel()) & (col_ids[inv.clamp(max=col_ids.numel() - 1)] == cols)
        rows, cols, vals, inv = rows[ok], cols[ok], vals[ok], inv[ok]
    p = int(col_ids.numel())
    sq = torch.zeros(p, dtype=torch.float64, device=dev).index_add_(0, inv, vals.double() ** 2)
    sm = torch.zeros(p, dtype=torch.float64, device=dev).index_add_(0, inv, vals.double())
    s = sq.sqrt()
    s_safe = torch.where(s > 0, s, torch.ones_like(s))
    sv = (vals.double() / s_safe[inv]).to(dtype)
    A = torch.sparse_coo_tensor(torch.stack([rows, inv]), sv, (n, p)).coalesce().to_sparse_csr()
    At = torch.sparse_coo_tensor(torch.stack([inv, rows]), sv, (p, n)).coalesce().to_sparse_csr()
    yf = y.to(dtype)
    ybar = float(yf.double().mean())
    return Design(A=A, At=At, m=(sm / s_safe / n).to(dtype), s=s.to(dtype), xm=(sm / n).to(dtype), y=yf, ybar=ybar,
                  yc=(yf.double() - ybar).to(dtype), col=col_ids, n=n, p=p)


def lam_max(D: Design) -> float:
    """The smallest lambda at which beta = 0 is optimal: max_j |A_c^T yc|_j / n."""
    return float(D.rmv(D.yc).abs().max()) / D.n if D.p else 0.0


def lipschitz(D: Design, iters: int = 60, seed: int = 0) -> float:
    """L = sigma_max(A_c)^2 / n by power iteration (+5% margin, so 1/L is a safe fixed step)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(D.p, generator=g).to(D.y.device, D.y.dtype)
    v = v / v.norm()
    lam = 0.0
    for _ in range(iters):
        w = D.rmv(D.mv(v)) / D.n
        lam = float(w.norm())
        if lam == 0:
            return 1e-12
        v = w / lam
    return 1.05 * lam


def soft(x, t):
    return torch.sign(x) * torch.clamp(x.abs() - t, min=0.0)


def objective(D: Design, beta, lam):
    r = D.yc - D.mv(beta)
    return float(0.5 / D.n * torch.dot(r, r)) + lam * float(beta.abs().sum())


def fista(D: Design, lam: float, beta0=None, L=None, tol: float = 1e-6, max_iter: int = 3000, min_iter: int = 5):
    """FISTA on the scaled problem at one lambda; returns (beta_scaled, info). Function-value restart: when an
    iterate raises the objective, the momentum is reset and the step is retaken from the last accepted point."""
    L = lipschitz(D) if L is None else L
    x = torch.zeros(D.p, dtype=D.y.dtype, device=D.y.device) if beta0 is None else beta0.clone()
    z, t = x.clone(), 1.0
    F = objective(D, x, lam)
    it, restarts, conv = 0, 0, False
    for it in range(1, max_iter + 1):
        g = -D.rmv(D.yc - D.mv(z)) / D.n
        xn = soft(z - g / L, lam / L)
        Fn = objective(D, xn, lam)
        if Fn > F:                                         # restart: drop the momentum, keep x
            restarts += 1
            if torch.equal(z, x):                          # a plain prox-gradient step went up: numerical floor
                conv = True
                break
            z, t = x.clone(), 1.0
            continue
        tn = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = xn + ((t - 1.0) / tn) * (xn - x)
        dec = (F - Fn) / max(abs(F), 1e-30)
        x, t, F = xn, tn, Fn
        if it >= min_iter and dec < tol:
            conv = True
            break
    return x, dict(iters=it, restarts=restarts, obj=F, converged=conv)


def original_units(D: Design, beta_scaled):
    """(beta in the raw X units, intercept b) from the scaled solution (C.1: 'coefficients are rescaled afterward')."""
    s = torch.where(D.s > 0, D.s, torch.ones_like(D.s))
    beta = beta_scaled / s
    b = D.ybar - float(torch.dot(D.xm.double(), beta.double()))
    return beta, b


def predict(D_or_rows, beta, b, n=None):
    """y_hat = b + X beta on raw COO triplets (rows, cols as Design column indices, vals) -- for held-out folds."""
    rows, cidx, vals = D_or_rows
    out = torch.full((int(n),), float(b), dtype=torch.float64, device=vals.device)
    return out.index_add_(0, rows.long(), (vals.double() * beta.double()[cidx.long()]))


def r2(y, yhat):
    y, yhat = y.double(), yhat.double()
    sst = float(((y - y.mean()) ** 2).sum())
    return float("nan") if sst == 0 else 1.0 - float(((y - yhat) ** 2).sum()) / sst


def lam_grid(lmax: float, per_decade: int = 20, min_ratio: float = 1e-4) -> List[float]:
    """lam_max * 10^(-k / per_decade), k = 1 .. per_decade * log10(1 / min_ratio) (lam_max itself has an empty support)."""
    k = int(round(per_decade * math.log10(1.0 / min_ratio)))
    return [lmax * 10 ** (-(j + 1) / per_decade) for j in range(k)]


def lasso_path(D: Design, lams: Sequence[float], on_step: Callable, tol: float = 1e-6, max_iter: int = 3000,
               support_cap: Optional[int] = None):
    """Warm-started path over `lams` (descending). on_step(k, lam, beta_scaled, info) -> True stops the path. Also stops
    when the support exceeds support_cap. Returns the list of per-step info dicts."""
    L = lipschitz(D)
    beta = torch.zeros(D.p, dtype=D.y.dtype, device=D.y.device)
    log = []
    for k, lam in enumerate(lams):
        t0 = time.time()
        beta, info = fista(D, lam, beta, L, tol=tol, max_iter=max_iter)
        info.update(k=k, lam=lam, nnz=int((beta != 0).sum()), secs=round(time.time() - t0, 3))
        log.append(info)
        if on_step(k, lam, beta, info):
            break
        if support_cap is not None and info["nnz"] >= support_cap:
            break
    return log


# ---------------------------------------------------------------------------------------------- capture
def _top(score, idx, n):
    """The n entries of idx with the largest score, ordered by score descending, then by column index (deterministic)."""
    if n <= 0 or idx.numel() == 0:
        return idx[:0]
    idx = idx[torch.argsort(idx, stable=True)]                  # column order first ...
    return idx[torch.argsort(-score[idx].double(), stable=True)][:n]   # ... then score, stable: ties by column


@dataclass
class PathCapture:
    """Collects, along a path, the member sets a ranked-members row needs.

    sizes   {name: n}       global sizes (e.g. {"W_0.001": 412, ...})
    counts  {site_j: c}     per-site counts (site index -> count; the DAN-43 matched counts)
    site_of [p] long        the site index of each Design column
    rankings                names of the scores to rank by: "coef" = |beta| in original units (the paper's |A|),
                            "std" = |beta_scaled| (the standardised coefficient)
    Each capture stores, per ranking, the selected column indices in rank order, plus the lambda, the step and the
    support size at which it was taken. A size the path never reached is filled at the end from the last step's
    support (all of it, marked short)."""
    sizes: Dict[str, int]
    counts: Dict[int, int]
    site_of: torch.Tensor
    s: torch.Tensor
    rankings: Sequence[str] = ("coef", "std")
    got_sizes: Dict[str, dict] = field(default_factory=dict)
    got_sites: Dict[int, dict] = field(default_factory=dict)
    last: Optional[dict] = None

    def scores(self, beta_scaled):
        s = torch.where(self.s > 0, self.s, torch.ones_like(self.s))
        return {"coef": (beta_scaled / s).abs(), "std": beta_scaled.abs()}

    def step(self, k, lam, beta_scaled):
        nz = (beta_scaled != 0).nonzero(as_tuple=True)[0]
        sc = None
        for name, n in self.sizes.items():
            if name not in self.got_sizes and nz.numel() >= n:
                sc = sc or self.scores(beta_scaled)
                self.got_sizes[name] = dict(lam=lam, k=k, support=int(nz.numel()), short=False,
                                            sel={r: _top(sc[r], nz, n) for r in self.rankings})
        if self.counts:
            site_nz = self.site_of[nz]
            for j, c in self.counts.items():
                if j in self.got_sites or c <= 0:
                    continue
                here = nz[site_nz == j]
                if here.numel() >= c:
                    sc = sc or self.scores(beta_scaled)
                    self.got_sites[j] = dict(lam=lam, k=k, support=int(here.numel()), short=False,
                                             sel={r: _top(sc[r], here, c) for r in self.rankings})
        self.last = dict(lam=lam, k=k, beta=beta_scaled.detach().clone())

    def done(self):
        return (all(n in self.got_sizes for n in self.sizes)
                and all(j in self.got_sites for j, c in self.counts.items() if c > 0))

    def finish(self):
        """Fill what the path never reached from the last step's support (all of it; short=True)."""
        if self.last is None:
            return
        beta = self.last["beta"]
        nz = (beta != 0).nonzero(as_tuple=True)[0]
        sc = self.scores(beta)
        for name, n in self.sizes.items():
            if name not in self.got_sizes:
                self.got_sizes[name] = dict(lam=self.last["lam"], k=self.last["k"], support=int(nz.numel()),
                                            short=True, sel={r: _top(sc[r], nz, n) for r in self.rankings})
        site_nz = self.site_of[nz]
        for j, c in self.counts.items():
            if c > 0 and j not in self.got_sites:
                here = nz[site_nz == j]
                self.got_sites[j] = dict(lam=self.last["lam"], k=self.last["k"], support=int(here.numel()),
                                         short=True, sel={r: _top(sc[r], here, c) for r in self.rankings})
