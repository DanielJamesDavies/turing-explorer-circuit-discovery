from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from pipeline.distributed.devices import (
    build_device_assignments,
    validate_worker_isolation,
    worker_environment,
    worker_local_devices,
)
from pipeline.distributed.manifest import DeviceAssignment
from pipeline.runtime import (
    build_distributed_worker_runtime,
    clear_runtime,
    initialize_resources,
    set_runtime,
)


def test_build_device_assignments_records_physical_identity(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda idx: SimpleNamespace(
            name=f"H100-{idx}",
            uuid=f"GPU-{idx}",
            pci_bus_id=f"0000:0{idx + 1}:00.0",
            total_memory=80 * 1024**3,
        ),
    )

    assignments = build_device_assignments(worker_count=2, physical_ids=[3, 1])

    assert [assignment.worker_id for assignment in assignments] == [0, 1]
    assert [assignment.physical_id for assignment in assignments] == [3, 1]
    assert [assignment.logical_id for assignment in assignments] == ["cuda:0", "cuda:0"]
    assert assignments[1].name == "H100-1"
    assert assignments[1].uuid == "GPU-1"
    assert assignments[1].pci_bus_id == "0000:02:00.0"
    assert assignments[1].total_vram_bytes == 80 * 1024**3


def test_build_device_assignments_rejects_oversubscription_by_default():
    with pytest.raises(ValueError, match="physical device IDs must be unique"):
        build_device_assignments(worker_count=2, physical_ids=[0, 0])


def test_cpu_fallback_is_single_worker_only():
    assignment = build_device_assignments(worker_count=1, use_cpu=True)[0]

    assert assignment.physical_id is None
    assert assignment.logical_id == "cpu"
    assert worker_local_devices(assignment) == [torch.device("cpu")]
    assert worker_environment(assignment) == {"CUDA_VISIBLE_DEVICES": ""}

    with pytest.raises(ValueError, match="CPU fallback supports exactly one worker"):
        build_device_assignments(worker_count=2, use_cpu=True)


def test_worker_environment_maps_physical_gpu_to_worker_local_cuda_zero():
    assignment = DeviceAssignment(worker_id=0, physical_id=7, logical_id="cuda:0")

    assert worker_environment(assignment) == {"CUDA_VISIBLE_DEVICES": "7"}
    assert worker_local_devices(assignment) == [torch.device("cuda:0")]
    validate_worker_isolation(assignment)


def test_device_assignment_rejects_non_isolated_logical_cuda_device():
    with pytest.raises(ValidationError, match="worker-local logical_id='cuda:0'"):
        DeviceAssignment(worker_id=0, physical_id=2, logical_id="cuda:2")


def test_device_assignment_rejects_cpu_worker_with_cuda_logical_id():
    with pytest.raises(ValidationError, match="CPU workers must use logical_id='cpu'"):
        DeviceAssignment(worker_id=0, physical_id=None, logical_id="cuda:0")


def test_distributed_worker_runtime_passes_one_device_to_sae_bank(monkeypatch):
    seen = {}

    class FakeDataLoader:
        def __init__(self, device, pin_memory):
            seen["loader_device"] = device
            seen["pin_memory"] = pin_memory
            self._shard_sequence_counts = [1, 2]

    class FakeSeqRepr:
        repr_dim = 4
        repr_mode = "mean_pool"

        def __init__(self, n_seqs):
            seen["n_seqs"] = n_seqs

    class FakeInference:
        def __init__(self, device, compile):
            seen["model_device"] = device
            seen["compile"] = compile

    class FakeSAEBank:
        def __init__(self, devices, load_decoders, compile):
            seen["sae_devices"] = devices
            seen["load_decoders"] = load_decoders

    monkeypatch.setattr("pipeline.runtime.DataLoader", FakeDataLoader)
    monkeypatch.setattr("pipeline.runtime.SeqRepr", FakeSeqRepr)
    monkeypatch.setattr("pipeline.runtime.Inference", FakeInference)
    monkeypatch.setattr("pipeline.runtime.SAEBank", FakeSAEBank)

    assignment = DeviceAssignment(worker_id=0, physical_id=5, logical_id="cuda:0")
    runtime = build_distributed_worker_runtime(assignment, apply_environment=False)
    set_runtime(runtime)
    try:
        initialize_resources()
    finally:
        clear_runtime()

    assert runtime.devices == [torch.device("cuda:0")]
    assert runtime.multi_gpu is False
    assert seen["loader_device"] == torch.device("cuda:0")
    assert seen["model_device"] == torch.device("cuda:0")
    assert seen["sae_devices"] == [torch.device("cuda:0")]
    assert len(seen["sae_devices"]) == 1
    assert seen["n_seqs"] == 3
