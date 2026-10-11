from types import SimpleNamespace

import pytest
import torch

from training.deepspeed_universal import (
    _attach_flat_parameters,
    _partition_tensor,
    _state_ids_in_order,
)


def test_grouped_universal_optimizer_mapping():
    optimizer_state = {
        "state": {0: {}, 1: {}, 2: {}},
        "param_groups": [{"params": [0, 1]}, {"params": [2]}],
    }
    fp32_groups = [torch.tensor([10]), torch.tensor([11]), torch.tensor([12])]
    optimizer = SimpleNamespace(param_groups=[{"params": []}, {"params": []}])

    assert _state_ids_in_order(optimizer_state, 3) == [0, 1, 2]
    _attach_flat_parameters(optimizer, fp32_groups, optimizer_state["param_groups"])
    assert optimizer.param_groups[0]["params"] == fp32_groups[:2]
    assert optimizer.param_groups[1]["params"] == fp32_groups[2:]


def test_grouped_universal_optimizer_mapping_rejects_mismatch():
    with pytest.raises(ValueError, match="optimizer states"):
        _state_ids_in_order({"state": {0: {}}}, 2)
    optimizer = SimpleNamespace(param_groups=[{"params": []}])
    with pytest.raises(ValueError, match="parameter groups"):
        _attach_flat_parameters(optimizer, [torch.tensor([1])], [{}, {}])


def test_universal_partition_uses_global_data_parallel_rank():
    tensor = torch.arange(18)
    assert _partition_tensor(tensor, world_size=16, rank=0).tolist() == [0, 1]
    assert _partition_tensor(tensor, world_size=16, rank=8).tolist() == [16, 17]
    assert _partition_tensor(tensor, world_size=16, rank=15).tolist() == [0, 0]
