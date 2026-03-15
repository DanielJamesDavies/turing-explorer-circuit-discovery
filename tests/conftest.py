"""
Shared test fixtures for the circuit interpretability test suite.
Expanded progressively through phases.

Phase 2: MockAttn, MockMLP, MockNorm2, MockBlock, MockModel, mock_model, make_circuit
Phase 3: MockSAEModule, MockSAEBank, mock_sae_bank, avg_acts_zero, avg_acts_nonzero
"""
import pytest
import torch
import torch.nn as nn

from store.circuits import Circuit, CircuitNode

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

D_MODEL = 16   # tiny embedding dimension — all mock tensors use this width
N_LAYERS = 2   # tiny transformer depth
D_SAE    = 32  # tiny SAE dictionary — must be > K_SAE
K_SAE    = 4   # top-k active features per token
KINDS    = ["attn", "mlp", "resid"]

# ---------------------------------------------------------------------------
# Phase 2 — Minimal model mock
#
# MockBlock.forward returns (attn_out, mlp_out, resid_out), exactly matching
# the turingllm.Block interface that hooks.py depends on.  The three fixed
# scale factors make expected values trivially computable in tests:
#
#   attn_out  = x * 0.1
#   mlp_out   = x * 0.2   (norm_2 is identity)
#   resid_out = x + x*0.1 + x*0.2 = x * 1.3
# ---------------------------------------------------------------------------

class MockAttn(nn.Module):
    """Scaled identity — produces a distinguishable attn-like output."""
    def forward(self, x):
        return x * 0.1


class MockMLP(nn.Module):
    """Scaled identity — produces a distinguishable mlp-like output."""
    def forward(self, x):
        return x * 0.2


class MockNorm2(nn.Module):
    """Identity normalisation (passthrough). Named norm_2 to match Block."""
    def forward(self, x):
        return x


class MockBlock(nn.Module):
    """
    Minimal transformer block with the same interface as turingllm.Block:
      - Submodules .attn, .mlp, .norm_2 (required by hooks.py)
      - forward(x) → (attn_out, mlp_out, resid_out) 3-tuple
    """
    def __init__(self, n_embd: int = D_MODEL):
        super().__init__()
        self.attn = MockAttn()
        self.mlp = MockMLP()
        self.norm_2 = MockNorm2()

    def forward(self, x, input_pos=None):
        attn_out = self.attn(x)
        mlp_out = self.mlp(self.norm_2(x))
        resid = x + attn_out + mlp_out
        return attn_out, mlp_out, resid


class MockModel(nn.Module):
    """
    Minimal model with the same structure as TuringLLM:
      - .transformer.h is a ModuleList of MockBlocks
      - forward(x) takes a pre-embedded float tensor [B, T, D_MODEL]
    """
    def __init__(self, n_layers: int = N_LAYERS, n_embd: int = D_MODEL):
        super().__init__()
        self.transformer = nn.ModuleDict({
            "h": nn.ModuleList([MockBlock(n_embd) for _ in range(n_layers)])
        })

    def forward(self, x):
        for block in self.transformer.h:
            _, _, x = block(x)
        return x


@pytest.fixture
def mock_model():
    """A fresh MockModel instance for each test."""
    return MockModel()


@pytest.fixture
def make_circuit():
    """
    Factory fixture for building test circuits from a compact spec.

    Usage:
        circuit = make_circuit([(layer_idx, kind, latent_idx, role), ...])
    """
    def _factory(nodes_spec):
        c = Circuit(name="test-circuit")
        for layer_idx, kind, latent_idx, role in nodes_spec:
            c.add_node(CircuitNode(metadata={
                "layer_idx": layer_idx,
                "kind": kind,
                "latent_idx": latent_idx,
                "role": role,
            }))
        return c
    return _factory


# ---------------------------------------------------------------------------
# Phase 3 — Minimal SAE mock
#
# MockSAEModule mirrors the interface that CircuitPatcher reads from
# bank.saes[kind][layer]:
#   .decoder_bias           tensor [d_model]
#   .encode(x)           -> (top_acts [B,T,k], top_indices [B,T,k])
#   .decode(latents)     -> x_hat [B,T,d_model]  (latents @ W_dec.T + decoder_bias)
#
# MockSAEBank mirrors the interface CircuitPatcher calls on the bank object:
#   .n_layer, .d_sae, .kinds, .layer_device_map, .saes[kind][layer]
#   .encode(x, kind, layer)      -> (top_acts, top_indices)
#   .decode(latents, kind, layer)-> x_hat
#
# Weights are initialised from a fixed seed so tests are deterministic.
# decoder_bias is non-zero so bias-cancellation arithmetic is exercised.
# ---------------------------------------------------------------------------

class MockSAEModule:
    """
    Lightweight SAE replacement backed by plain torch tensors.

    Encoder:  pre = ReLU(x @ W_enc.T + b_enc), returns top-k of pre.
    Decoder:  latents @ W_dec.T + decoder_bias
    """
    def __init__(self, d_model: int, d_sae: int, k: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc       = torch.randn(d_sae, d_model, generator=g)
        self.b_enc       = torch.zeros(d_sae)
        self.W_dec       = torch.randn(d_model, d_sae, generator=g)
        # Non-zero so that bias double-counting tests are meaningful.
        self.decoder_bias = torch.randn(d_model, generator=g) * 0.1

    def encode(self, x: torch.Tensor):
        """Returns (top_acts [*,k], top_indices [*,k]) where * = x.shape[:-1]."""
        pre = torch.relu(x.float() @ self.W_enc.T + self.b_enc)
        return pre.topk(self.k, dim=-1)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents [*, d_sae] -> x_hat [*, d_model]."""
        return latents.float() @ self.W_dec.T + self.decoder_bias


class MockSAEBank:
    """
    Lightweight SAEBank replacement with the exact attribute/method surface
    that CircuitPatcher reads.  One MockSAEModule per (kind, layer) pair.
    """
    def __init__(
        self,
        n_layers: int = N_LAYERS,
        d_model:  int = D_MODEL,
        d_sae:    int = D_SAE,
        k:        int = K_SAE,
        kinds           = None,
    ):
        if kinds is None:
            kinds = KINDS
        self.n_layer         = n_layers
        self.d_model         = d_model
        self.d_sae           = d_sae
        self.k               = k
        self.kinds           = kinds
        self.device          = torch.device("cpu")
        self.layer_device_map = {l: torch.device("cpu") for l in range(n_layers)}

        # saes[kind][layer] follows the same nested structure as SAEBank.saes
        self.saes = {
            kind: [
                MockSAEModule(d_model, d_sae, k, seed=l * len(kinds) + i)
                for l in range(n_layers)
            ]
            for i, kind in enumerate(kinds)
        }

    def encode(self, x: torch.Tensor, kind: str, layer: int):
        return self.saes[kind][layer].encode(x)

    def decode(self, latents: torch.Tensor, kind: str, layer: int) -> torch.Tensor:
        return self.saes[kind][layer].decode(latents)


@pytest.fixture
def mock_sae_bank():
    """A fresh MockSAEBank for each test."""
    return MockSAEBank()


@pytest.fixture
def avg_acts_zero():
    """All-zero background activations [n_components, D_SAE]."""
    return torch.zeros(N_LAYERS * len(KINDS), D_SAE)


@pytest.fixture
def avg_acts_nonzero():
    """
    Non-zero background activations [n_components, D_SAE].
    Each latent j has a small, distinct background value 0.01*(j+1)
    so that bias-subtraction and background-masking tests are non-trivial.
    """
    acts = torch.zeros(N_LAYERS * len(KINDS), D_SAE)
    for j in range(D_SAE):
        acts[:, j] = 0.01 * (j + 1)
    return acts
