"""Discovery method ownership helpers for distributed workers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Sequence

from config import config

from ..manifest import DistributedRunManifest


SEED_FREE_DISCOVERY_METHODS = {"cluster_contrast"}


def seed_free_methods_for_worker(
    manifest: DistributedRunManifest,
    worker_id: int,
) -> List[str]:
    """Return seed-free methods owned by this worker."""

    return sorted(
        method
        for method, owner in manifest.work_assignments.discovery_seed_free_method_owners.items()
        if owner == worker_id
    )


def discovery_methods_for_worker_filter(
    methods: Sequence[str],
    seed_free_methods: Sequence[str],
) -> List[str]:
    """Filter seed-free methods so only explicitly owned ones remain enabled."""

    allowed_seed_free = set(seed_free_methods)
    return [
        str(method)
        for method in methods
        if str(method) not in SEED_FREE_DISCOVERY_METHODS or str(method) in allowed_seed_free
    ]


@contextmanager
def discovery_methods_for_worker(seed_free_methods: Sequence[str]):
    """Temporarily filter global discovery methods for one worker run."""

    original_methods = config.discovery.methods
    config.discovery.methods = discovery_methods_for_worker_filter(
        list(original_methods),
        seed_free_methods,
    )
    try:
        yield
    finally:
        config.discovery.methods = original_methods


__all__ = [
    "SEED_FREE_DISCOVERY_METHODS",
    "discovery_methods_for_worker",
    "discovery_methods_for_worker_filter",
    "seed_free_methods_for_worker",
]
