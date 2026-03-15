from typing import Any, Dict, List, cast

import torch

from circuit.feature_selection import CandidateSelector
from config import config


def run_candidate_selection() -> List[Dict[str, Any]]:
    print("--- Candidate Selection: Finding Seeds ---")
    n_seeds = cast(int, config.discovery.n_seeds or 1000)
    selector = CandidateSelector(n_seeds=n_seeds)
    candidates = selector.select_candidates()
    selector.get_summary_stats(candidates)

    torch.save(candidates, "outputs/candidates.pt")
    print("  ✓ candidates saved to outputs/candidates.pt")
    print("")
    return candidates
