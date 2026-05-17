import time
from typing import Any, Dict, List, cast

import torch

from circuit.feature_selection import CandidateSelector
from config import config
from observability.timing import format_duration


def run_candidate_selection() -> List[Dict[str, Any]]:
    print("--- Candidate Selection: Finding Seeds ---")
    n_seeds = cast(int, config.discovery.n_seeds or 1000)
    selector = CandidateSelector(n_seeds=n_seeds)
    select_t0 = time.perf_counter()
    candidates = selector.select_candidates()
    print(f"  [timing] candidate scoring: {format_duration(time.perf_counter() - select_t0)}")
    selector.get_summary_stats(candidates)

    save_t0 = time.perf_counter()
    torch.save(candidates, "outputs/candidates.pt")
    print(f"  [timing] candidates save: {format_duration(time.perf_counter() - save_t0)}")
    print("  ✓ candidates saved to outputs/candidates.pt")
    print("")
    return candidates
