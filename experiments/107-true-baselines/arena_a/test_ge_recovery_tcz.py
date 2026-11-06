"""CPU unit tests for ge_recovery_tcz.py (ge107, no model): job nodes -> Ge's node keys and masks.
  source ~/venvs/ge107/bin/ge_env.sh; python -I -B -m pytest -q -p no:cacheprovider \
      experiments/107-true-baselines/arena_a/test_ge_recovery_tcz.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import ge_core as GC  # noqa: E402
import ge_recovery_tcz as GR  # noqa: E402

HP_Z = "blocks.5.attn.hook_z.sae.hook_feature_acts"
HP_TC = "blocks.2.hook_mlp_out.sae.hook_feature_acts"


def test_node_set():
    job = dict(nodes=[[HP_Z, 1, 24577], [HP_TC, 3, 9], [HP_TC, 3, 9]])
    assert GR.node_set(job) == {(HP_Z, "0.1.24577"), (HP_TC, "0.3.9")}


def test_masks_keep_nodes_and_root():
    from lm_saes.circuit.graph import Node
    st = dict(acts={HP_Z: torch.zeros(1, 4, 49152), HP_TC: torch.zeros(1, 4, 24576),
                    "blocks.6.hook_mlp_out.sae.hook_feature_acts": torch.zeros(1, 4, 24576)})
    target = Node("blocks.6.hook_mlp_out.sae.hook_feature_acts", "0.3.7")
    m = GC.masks_from_nodes(GR.node_set(dict(nodes=[[HP_Z, 1, 24577], [HP_TC, 3, 9]])), st, target)
    assert float(m[HP_Z].sum()) == 1 and m[HP_Z][0, 1, 24577] == 1
    assert float(m[HP_TC].sum()) == 1 and m[HP_TC][0, 3, 9] == 1
    assert float(m[target.hook_point].sum()) == 1 and m[target.hook_point][0, 3, 7] == 1     # root kept (A8)
    # WCM rows (pos None): every non-BOS position 1..T-1, never BOS
    m = GR.masks_for(dict(nodes=[[HP_TC, None, 9], [HP_Z, 2, 5]]), st, target)
    assert m[HP_TC][0, :, 9].tolist() == [0, 1, 1, 1] and float(m[HP_TC].sum()) == 3
    assert float(m[HP_Z].sum()) == 1 and m[HP_Z][0, 2, 5] == 1
