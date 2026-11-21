"""Donor-code trim (2026-11-18, exp 128 RAM probe): capture_donor_codes drops slots that are 0 at every position, so
ReLU dictionaries run at k = d_sae no longer store every slot. CPU only. What must hold:
  - the DENSIFIED donor code (what DonorPool.get hands the engine) is bit-identical to the untrimmed one, for codes
    with many zero slots (k = d_sae, the arena A case), for batches trimmed to different widths (the padded case),
    and for Top-K codes with no zero slot (untouched: same tensors, scatter_ path);
  - a fit with the resample term is bit-identical with and without the trim (same circuit, same loss curve).
"""
import torch

from circuit.instrument import learned_mask as LM
from circuit.instrument.learned_mask import DonorPool, capture_donor_codes

D_MODEL, D_SAE = 16, 64
SITE = (0, "resid")


class ReluBank:
    """ReLU dictionary stored Top-K-style with k = d_sae (arena A): encode = topk(relu(x W^T + b), d_sae).
    A negative bias makes most slots exactly 0; `sparsity` sets how many."""
    d_sae = D_SAE
    kinds = ["attn", "mlp", "resid"]
    device = torch.device("cpu")

    def __init__(self, k=D_SAE, bias=-1.5, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.k = k
        self.W = torch.randn(D_SAE, D_MODEL, generator=g)
        self.b = torch.full((D_SAE,), float(bias))

    def encode(self, x, kind, layer):
        return torch.relu(x.float() @ self.W.T + self.b).topk(self.k, dim=-1)


class FakeInference:
    def __init__(self, x):
        self.x = x

    def disable_compile(self):
        pass

    def enable_compile(self):
        pass

    def forward(self, tokens, activations_callback=None, **kw):
        x = self.x[tokens[:, 0].long()].clone()
        activations_callback(0, (x, x, x))


def _setup(n=12, T=5, scale=1.0, seed=7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, T, D_MODEL, generator=g) * scale
    tok = torch.arange(n)[:, None].repeat(1, T)
    return FakeInference(x), tok


def _untrimmed(inf, bank, tok, batch):
    """The pre-2026-11-18 capture: every slot kept."""
    A, I = [], []
    for s0 in range(0, tok.shape[0], batch):
        a, i = bank.encode(inf.x[tok[s0:s0 + batch, 0]], "resid", 0)
        A.append(a.float())
        I.append(i)
    return torch.cat(A), torch.cat(I)


def _dense_all(codes, n, device=torch.device("cpu")):
    pool = DonorPool(codes, D_SAE, device, seed=5)
    pool.rows = torch.arange(n)
    return pool.get(SITE)


def _dense_ref(a, i):
    return torch.zeros(a.shape[0], a.shape[1], D_SAE).scatter_(-1, i, a)


def test_relu_k_dsae_trimmed_and_identical():
    inf, tok = _setup()
    bank = ReluBank()
    codes = capture_donor_codes(inf, bank, tok, [SITE], batch=4)
    a, i = codes[SITE]
    ra, ri = _untrimmed(inf, bank, tok, 4)
    assert a.shape[-1] < ra.shape[-1], "a ReLU code at k = d_sae must be trimmed"
    assert int((ra != 0).sum(-1).max()) <= a.shape[-1]
    assert torch.equal(_dense_all(codes, tok.shape[0]), _dense_ref(ra, ri))


def test_batches_of_different_widths_pad_and_stay_identical():
    # rows 0-3 nearly silent, rows 4-11 busy: the first batch is trimmed narrower than the rest -> padded
    inf, tok = _setup()
    inf.x[:4] *= 0.05
    bank = ReluBank()
    codes = capture_donor_codes(inf, bank, tok, [SITE], batch=4)
    a, i = codes[SITE]
    pool = DonorPool(codes, D_SAE, torch.device("cpu"))
    assert SITE in pool.padded, "this case must take the padded path"
    ra, ri = _untrimmed(inf, bank, tok, 4)
    assert torch.equal(_dense_all(codes, tok.shape[0]), _dense_ref(ra, ri))
    # random draws too (the engine's access pattern)
    pool.draw(9)
    assert torch.equal(pool.get(SITE), _dense_ref(ra, ri)[pool.rows])


def test_all_zero_rows_kept_as_one_slot():
    inf, tok = _setup()
    inf.x[:] = 0.0                                                  # every code exactly 0 (bias < 0)
    bank = ReluBank()
    codes = capture_donor_codes(inf, bank, tok, [SITE], batch=4)
    a, i = codes[SITE]
    assert a.shape[-1] == 1 and not a.any()
    assert torch.equal(_dense_all(codes, tok.shape[0]), torch.zeros(tok.shape[0], tok.shape[1], D_SAE))


def test_topk_without_zero_slots_untouched():
    inf, tok = _setup(scale=3.0)
    bank = ReluBank(k=4, bias=10.0)                                 # every top-4 slot positive
    codes = capture_donor_codes(inf, bank, tok, [SITE], batch=4)
    a, i = codes[SITE]
    ra, ri = _untrimmed(inf, bank, tok, 4)
    assert torch.equal(a, ra) and torch.equal(i, ri)
    assert SITE not in DonorPool(codes, D_SAE, torch.device("cpu")).padded


def test_unsorted_codes_keep_every_nonzero():
    # an encode that returns its slots unsorted (zeros interleaved): the trim must still keep every nonzero slot
    inf, tok = _setup()
    bank = ReluBank()
    base_encode = bank.encode

    def shuffled(x, kind, layer):
        a, i = base_encode(x, kind, layer)
        p = torch.randperm(a.shape[-1], generator=torch.Generator().manual_seed(1))
        return a[..., p], i[..., p]

    bank.encode = shuffled
    codes = capture_donor_codes(inf, bank, tok, [SITE], batch=4)
    ra, ri = _untrimmed(inf, bank, tok, 4)
    assert torch.equal(_dense_all(codes, tok.shape[0]), _dense_ref(ra, ri))


def test_memory_shrinks_for_sparse_relu():
    inf, tok = _setup(n=16)
    bank = ReluBank(bias=-3.0)
    a, i = capture_donor_codes(inf, bank, tok, [SITE], batch=4)[SITE]
    ra, ri = _untrimmed(inf, bank, tok, 4)
    full = ra.numel() * ra.element_size() + ri.numel() * ri.element_size()
    kept = a.numel() * a.element_size() + i.numel() * i.element_size()
    assert kept < 0.5 * full
