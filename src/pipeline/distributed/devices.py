"""Device assignment and worker-local isolation helpers for distributed runs."""

from __future__ import annotations

import os
import socket
from typing import Dict, Iterable, List, Optional

import torch

from .manifest import DeviceAssignment


def _optional_str(value: object) -> Optional[str]:
    """Normalize best-effort CUDA metadata for strict manifest schemas."""

    if value is None:
        return None
    return str(value)


def build_device_assignments(
    worker_count: int,
    *,
    physical_ids: Optional[Iterable[int]] = None,
    visible_device_count: Optional[int] = None,
    use_cpu: bool = False,
    allow_oversubscription: bool = False,
) -> List[DeviceAssignment]:
    """Create one physical-device assignment per worker.

    CUDA workers always see their assigned physical device as worker-local
    ``cuda:0`` after launch-time environment isolation.
    """

    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    if visible_device_count is not None and visible_device_count < 0:
        raise ValueError("visible_device_count must be >= 0")
    if use_cpu:
        if worker_count != 1:
            raise ValueError("CPU fallback supports exactly one worker")
        return [
            DeviceAssignment(
                worker_id=0,
                physical_id=None,
                logical_id="cpu",
                hostname=socket.gethostname(),
            )
        ]

    selected_physical_ids = (
        list(physical_ids)
        if physical_ids is not None
        else list(range(worker_count))
    )
    if len(selected_physical_ids) != worker_count:
        raise ValueError("physical_ids length must match worker_count")
    if any(physical_id < 0 for physical_id in selected_physical_ids):
        raise ValueError("physical device IDs must be >= 0")
    if visible_device_count is not None:
        invalid_ids = [
            physical_id
            for physical_id in selected_physical_ids
            if physical_id >= visible_device_count
        ]
        if invalid_ids:
            raise ValueError(
                f"physical device IDs not visible: {sorted(set(invalid_ids))}"
            )
    if not allow_oversubscription and len(selected_physical_ids) != len(
        set(selected_physical_ids)
    ):
        raise ValueError("physical device IDs must be unique")

    return [
        collect_device_assignment(worker_id=worker_id, physical_id=physical_id)
        for worker_id, physical_id in enumerate(selected_physical_ids)
    ]


def collect_device_assignment(worker_id: int, physical_id: int) -> DeviceAssignment:
    """Collect best-effort physical GPU identity for the manifest."""

    if worker_id < 0:
        raise ValueError("worker_id must be >= 0")
    if physical_id < 0:
        raise ValueError("physical_id must be >= 0")

    name: Optional[str] = None
    uuid: Optional[str] = None
    pci_bus_id: Optional[str] = None
    total_vram_bytes: Optional[int] = None
    if torch.cuda.is_available() and physical_id < torch.cuda.device_count():
        props = torch.cuda.get_device_properties(physical_id)
        name = _optional_str(getattr(props, "name", None))
        uuid = _optional_str(getattr(props, "uuid", None))
        pci_bus_id = _optional_str(getattr(props, "pci_bus_id", None))
        total_vram_bytes = getattr(props, "total_memory", None)

    return DeviceAssignment(
        worker_id=worker_id,
        physical_id=physical_id,
        logical_id="cuda:0",
        uuid=uuid,
        name=name,
        pci_bus_id=pci_bus_id,
        total_vram_bytes=total_vram_bytes,
        hostname=socket.gethostname(),
    )


def worker_environment(assignment: DeviceAssignment) -> Dict[str, str]:
    """Return environment overrides that isolate one worker process."""

    if assignment.physical_id is None:
        return {"CUDA_VISIBLE_DEVICES": ""}
    return {"CUDA_VISIBLE_DEVICES": str(assignment.physical_id)}


def apply_worker_environment(assignment: DeviceAssignment) -> None:
    """Apply worker device isolation to the current process environment."""

    os.environ.update(worker_environment(assignment))


def worker_local_devices(assignment: DeviceAssignment) -> List[torch.device]:
    """Return the only devices a distributed worker is allowed to use."""

    if assignment.physical_id is None or assignment.logical_id == "cpu":
        return [torch.device("cpu")]
    return [torch.device("cuda:0")]


def validate_worker_isolation(assignment: DeviceAssignment) -> None:
    """Assert a worker assignment maps to exactly one worker-local device."""

    devices = worker_local_devices(assignment)
    if len(devices) != 1:
        raise ValueError("distributed workers must receive exactly one device")
    expected = "cpu" if assignment.physical_id is None else "cuda:0"
    if str(devices[0]) != expected:
        raise ValueError(f"worker-local device must be {expected}")
