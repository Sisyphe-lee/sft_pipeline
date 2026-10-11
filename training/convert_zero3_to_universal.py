"""Convert a multi-optimizer-group ZeRO-3 checkpoint to DeepSpeed universal format.

DeepSpeed 0.17.2's bundled converter flattens ``param_shapes`` across optimizer
groups, but reads only the first flattened optimizer tensor. This wrapper keeps
the groups aligned during extraction and delegates the rest of the format to
DeepSpeed's converter.
"""

from functools import partial
import math
import os

import torch


def _partitioned_numel(numel, world_size):
    return math.ceil(numel / world_size)


def _ordered_optimizer_states(optimizer_state, expected_groups):
    states = optimizer_state["state"]
    state_keys = list(states)
    if all(isinstance(key, int) for key in state_keys):
        state_keys.sort()
    ordered = [states[key] for key in state_keys]
    if len(ordered) != expected_groups:
        raise ValueError(
            f"Checkpoint has {len(ordered)} optimizer state groups but "
            f"{expected_groups} parameter-shape groups"
        )
    return ordered


def extract_grouped_zero3_shard(
    state_dict,
    param_shape_groups,
    dp_degree,
    dp_index,
    save_fragment,
):
    """Extract one data-parallel shard while preserving optimizer groups."""
    zero_state = state_dict["optimizer_state_dict"]
    fp32_groups = zero_state["fp32_flat_groups"]
    optimizer_states = _ordered_optimizer_states(
        zero_state["optimizer_state_dict"], len(param_shape_groups)
    )
    if len(fp32_groups) != len(param_shape_groups):
        raise ValueError(
            f"Checkpoint has {len(fp32_groups)} fp32 groups but "
            f"{len(param_shape_groups)} parameter-shape groups"
        )

    for group_index, (param_shapes, fp32, optimizer_group) in enumerate(
        zip(param_shape_groups, fp32_groups, optimizer_states, strict=True)
    ):
        flat_states = {
            "exp_avg": optimizer_group["exp_avg"],
            "exp_avg_sq": optimizer_group["exp_avg_sq"],
            "fp32": fp32,
        }
        offset = 0
        for name, shape in param_shapes.items():
            unpartitioned_numel = shape.numel()
            partitioned_numel = _partitioned_numel(unpartitioned_numel, dp_degree)
            padding_free_numel = min(
                partitioned_numel,
                max(0, unpartitioned_numel - dp_index * partitioned_numel),
            )
            for state_name, flat_tensor in flat_states.items():
                fragment = flat_tensor.narrow(0, offset, padding_free_numel).clone()
                save_fragment(name, state_name, dp_index, fragment)
            offset += partitioned_numel

        lengths = {state_name: tensor.numel() for state_name, tensor in flat_states.items()}
        if any(length != offset for length in lengths.values()):
            raise ValueError(
                f"Optimizer group {group_index} flattened length mismatch: "
                f"expected {offset}, found {lengths}"
            )


def _extract_shard(param_shape_groups, optim_files, dp_degree, temp_dir, dp_index):
    from deepspeed.checkpoint.ds_to_universal import _save_checkpoint

    state_dict = torch.load(optim_files[dp_index], map_location="cpu", weights_only=False)

    def save_fragment(param_name, state_name, rank, fragment):
        path = os.path.join(temp_dir, param_name, "0", f"{state_name}.{rank:02d}")
        _save_checkpoint(path, fragment)

    extract_grouped_zero3_shard(
        state_dict,
        param_shape_groups,
        dp_degree,
        dp_index,
        save_fragment,
    )


def _extract_all_shards(args, optim_files, _flat_param_shapes, dp_degree, temp_dir):
    from deepspeed.checkpoint import PARAM_SHAPES
    from deepspeed.checkpoint.ds_to_universal import _do_parallel_work, _get_model_state_files

    model_file = _get_model_state_files(args.input_folder)[0]
    param_shape_groups = torch.load(
        model_file, map_location="cpu", weights_only=False
    )[PARAM_SHAPES]
    work = partial(_extract_shard, param_shape_groups, optim_files, dp_degree, temp_dir)
    _do_parallel_work(work, list(range(dp_degree)), args.num_extract_workers)


def main():
    from deepspeed.checkpoint import ds_to_universal

    args = ds_to_universal.parse_arguments()
    ds_to_universal._extract_zero_shard_files_stage3 = _extract_all_shards
    ds_to_universal.main(args)


if __name__ == "__main__":
    main()
