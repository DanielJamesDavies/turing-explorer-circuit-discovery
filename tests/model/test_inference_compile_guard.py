from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import torch

from model.inference import Inference


class _TinyModel:
    def __init__(self):
        self.transformer = SimpleNamespace(h=[])

    def __call__(self, tokens: torch.Tensor, return_all_logits: bool = False):
        batch, seq_len = tokens.shape
        logits = torch.zeros(batch, seq_len, 8, dtype=torch.float32)
        logits[..., 0] = 1.0
        return logits, None


class _NoopPatcher:
    def __init__(self):
        self.entered = False

    def __call__(self, model):
        @contextmanager
        def _ctx():
            self.entered = True
            yield

        return _ctx()


def _fake_inference(compiled: bool = True):
    inference = SimpleNamespace()
    inference.model = _TinyModel()
    inference.device = torch.device("cpu")
    inference._compiled = compiled
    inference.events = []

    def disable_compile():
        inference.events.append("disable")
        inference._compiled = False

    def enable_compile():
        inference.events.append("enable")
        inference._compiled = True

    inference.disable_compile = disable_compile
    inference.enable_compile = enable_compile
    inference.forward = MethodType(Inference.forward, inference)
    return inference


def test_patcher_forward_temporarily_disables_and_restores_compile():
    inference = _fake_inference(compiled=True)
    patcher = _NoopPatcher()
    tokens = torch.zeros(2, 4, dtype=torch.long)

    _, logits, _ = inference.forward(tokens, patcher=patcher, return_activations=False, tokenize_final=False)

    assert patcher.entered is True
    assert logits.shape == (2, 4, 8)
    assert inference.events == ["disable", "enable"]
    assert inference._compiled is True


def test_uncompiled_patcher_forward_does_not_enable_compile_afterward():
    inference = _fake_inference(compiled=False)
    patcher = _NoopPatcher()
    tokens = torch.zeros(1, 3, dtype=torch.long)

    inference.forward(tokens, patcher=patcher, return_activations=False, tokenize_final=False)

    assert patcher.entered is True
    assert inference.events == []
    assert inference._compiled is False
