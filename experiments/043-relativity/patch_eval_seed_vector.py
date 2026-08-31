"""Teach circuit_only_activation / CircuitOnlyPatcher to read a VIRTUAL
seed direction (seed_vector=(w, b)) in the pre-activation frame, for
scoring differential-seed circuits. Post-top-k reads keep requiring a
real latent (a virtual direction has no top-k slot to be censored by).
Run from repo root.
"""
import ast

P = "src/eval/ablation_faithfulness.py"
s = open(P, encoding="utf-8").read()

# 1. patcher: accept + store
old = "        keep_scale: float = 1.0,\n        capture_preact: bool = False,"
assert s.count(old) == 1
s = s.replace(old, old + "\n        seed_vector=None,")
old = "        self.capture_preact = bool(capture_preact)"
assert s.count(old) == 1
s = s.replace(old, old + "\n        self.seed_vector = seed_vector")

# 2. capture: use the virtual direction when provided
old = ("                sae_seed = self.bank.saes[kind][layer_idx]\n"
       "                w = sae_seed.encoder.weight[self.seed_latent_idx]"
       ".detach().to(\n"
       "                    device=x.device, dtype=x.dtype)\n"
       "                b = sae_seed._get_bias_eff()[self.seed_latent_idx]"
       ".detach().to(\n"
       "                    device=x.device, dtype=x.dtype)")
assert s.count(old) == 1, "capture anchor not found"
s = s.replace(old, (
    "                if self.seed_vector is not None:\n"
    "                    w = self.seed_vector[0].detach().to(\n"
    "                        device=x.device, dtype=x.dtype)\n"
    "                    b = self.seed_vector[1].detach().to(\n"
    "                        device=x.device, dtype=x.dtype)\n"
    "                else:\n"
    "                    sae_seed = self.bank.saes[kind][layer_idx]\n"
    "                    w = sae_seed.encoder.weight[self.seed_latent_idx]"
    ".detach().to(\n"
    "                        device=x.device, dtype=x.dtype)\n"
    "                    b = sae_seed._get_bias_eff()[self.seed_latent_idx]"
    ".detach().to(\n"
    "                        device=x.device, dtype=x.dtype)"))

# 3. top-level function: accept + forward
old = "    keep_scales: \"Optional[Dict[Tuple[int, str], torch.Tensor]]\" = None,\n    preact: bool = False,"
assert s.count(old) == 1
s = s.replace(old, old + "\n    seed_vector=None,")
old = "                capture_preact=preact,"
assert s.count(old) == 1
s = s.replace(old, old + "\n                seed_vector=seed_vector,")

open(P, "w", encoding="utf-8", newline="").write(s)
ast.parse(s)
print("evaluator: seed_vector support added")
