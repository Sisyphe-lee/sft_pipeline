from collections import OrderedDict

import torch

from training.convert_zero3_to_universal import extract_grouped_zero3_shard


def test_extract_grouped_zero3_shard_resets_offsets_between_groups():
    param_shapes = [
        OrderedDict(first=torch.Size([5])),
        OrderedDict(second=torch.Size([3]), third=torch.Size([2])),
    ]
    # With DP=2, the flattened per-rank group sizes are 3 and 3.
    state = {
        "optimizer_state_dict": {
            "fp32_flat_groups": [torch.tensor([10, 11, 12]), torch.tensor([20, 21, 22])],
            "optimizer_state_dict": {
                "state": {
                    0: {
                        "exp_avg": torch.tensor([110, 111, 112]),
                        "exp_avg_sq": torch.tensor([210, 211, 212]),
                    },
                    1: {
                        "exp_avg": torch.tensor([120, 121, 122]),
                        "exp_avg_sq": torch.tensor([220, 221, 222]),
                    },
                },
                "param_groups": [{"params": [0]}, {"params": [1]}],
            },
        }
    }
    saved = {}

    extract_grouped_zero3_shard(
        state,
        param_shapes,
        dp_degree=2,
        dp_index=1,
        save_fragment=lambda name, state_name, rank, value: saved.setdefault(
            (name, state_name, rank), value
        ),
    )

    assert saved[("first", "fp32", 1)].tolist() == [10, 11]
    assert saved[("second", "fp32", 1)].tolist() == [20]
    assert saved[("third", "fp32", 1)].tolist() == [22]
    assert saved[("second", "exp_avg", 1)].tolist() == [120]
