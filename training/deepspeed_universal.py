"""Narrow DeepSpeed universal-checkpoint compatibility for grouped ZeRO-3 state."""

import math
import os


def _partition_tensor(tensor, world_size, rank):
    partitioned_numel = math.ceil(tensor.numel() / world_size)
    padded_numel = world_size * partitioned_numel
    if padded_numel != tensor.numel():
        import torch

        tensor = torch.cat(
            [
                tensor,
                torch.zeros(padded_numel - tensor.numel(), dtype=tensor.dtype),
            ]
        )
    return tensor.narrow(0, rank * partitioned_numel, partitioned_numel)


def _state_ids_in_order(optimizer_state, expected_groups):
    state_ids = list(optimizer_state["state"])
    if all(isinstance(state_id, int) for state_id in state_ids):
        state_ids.sort()
    if len(state_ids) != expected_groups:
        raise ValueError(
            f"Universal checkpoint has {len(state_ids)} optimizer states but "
            f"the current ZeRO-3 optimizer has {expected_groups} flattened groups"
        )
    return state_ids


def _attach_flat_parameters(optimizer, fp32_groups, saved_param_groups):
    if len(optimizer.param_groups) != len(saved_param_groups):
        raise ValueError(
            f"Universal checkpoint has {len(saved_param_groups)} optimizer parameter "
            f"groups but the current optimizer has {len(optimizer.param_groups)}"
        )
    for current_group, saved_group in zip(
        optimizer.param_groups, saved_param_groups, strict=True
    ):
        parameter_ids = saved_group["params"]
        if any(not isinstance(parameter_id, int) for parameter_id in parameter_ids):
            raise ValueError("Universal checkpoint optimizer parameter IDs must be integers")
        try:
            current_group["params"] = [
                fp32_groups[parameter_id] for parameter_id in parameter_ids
            ]
        except IndexError as error:
            raise ValueError(
                "Universal checkpoint optimizer parameter ID is out of range"
            ) from error


def enable_grouped_zero3_universal_load():
    """Patch DeepSpeed 0.17's single-group universal ZeRO-3 loader."""
    import torch
    from deepspeed import comm as dist
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

    if getattr(
        DeepSpeedZeroOptimizer_Stage3,
        "_physicalagent_grouped_universal_load",
        False,
    ):
        return

    def load_grouped_checkpoint(self, checkpoint_dir, param_shapes):
        zero_dir = os.path.join(checkpoint_dir, "zero")
        optimizer_path = os.path.join(zero_dir, "optimizer_state.pt")
        if not os.path.isfile(optimizer_path):
            raise FileNotFoundError(
                f"{optimizer_path} containing optimizer global state is missing"
            )

        universal_state = torch.load(optimizer_path, weights_only=False)
        self._load_global_state_stage3(universal_state)
        optimizer_state = universal_state["optimizer_state_dict"]
        state_ids = _state_ids_in_order(optimizer_state, len(param_shapes))
        if len(self.fp32_partitioned_groups_flat) != len(param_shapes):
            raise ValueError(
                f"Current ZeRO-3 optimizer has {len(self.fp32_partitioned_groups_flat)} "
                f"fp32 groups but {len(param_shapes)} parameter-shape groups"
            )

        def load_partition(layer, key):
            loaded = torch.load(
                os.path.join(zero_dir, layer, f"{key}.pt"),
                weights_only=False,
            ).view(-1)
            world_size = dist.get_world_size(group=self.dp_process_group)
            rank = dist.get_rank(group=self.dp_process_group)
            return _partition_tensor(loaded, world_size, rank)

        for group_index, (state_id, group_shapes) in enumerate(
            zip(state_ids, param_shapes, strict=True)
        ):
            for key in ("fp32", "exp_avg", "exp_avg_sq"):
                partitions = [
                    load_partition(layer, key)
                    for layer in group_shapes
                ]
                key_tensor = torch.cat(partitions) if partitions else torch.empty(0)
                if key == "fp32":
                    self.fp32_partitioned_groups_flat[group_index].data.copy_(key_tensor)
                else:
                    optimizer_state["state"][state_id][key] = key_tensor

        _attach_flat_parameters(
            self.optimizer,
            self.fp32_partitioned_groups_flat,
            optimizer_state["param_groups"],
        )

        if self.swap_optimizer:
            self.optimizer_swapper.purge_state()

        self.optimizer.load_state_dict(optimizer_state)
        for param_group in self.optimizer.param_groups:
            param_group["params"] = []

        for group_index, fp32_param in enumerate(self.fp32_partitioned_groups_flat):
            if sum(fp32_param.size()) > 0:
                fp16_param = self.fp16_partitioned_groups_flat[group_index]
                fp16_param.data.copy_(fp32_param.data)

        for group_index, fp16_group in enumerate(self.fp16_partitioned_groups_flat):
            updated = self.unflatten(fp16_group, self.fp16_partitioned_groups[group_index])
            for partitioned_param, value in zip(
                self.fp16_partitioned_groups[group_index], updated, strict=True
            ):
                partitioned_param.data = value.data

        if self.swap_optimizer:
            timer_names = set()
            self._partition_all_parameters()
            for group_index, _group in enumerate(self.fp16_groups):
                self._prepare_sub_group(group_index, timer_names)
                self._reassign_or_swap_out_partitioned_parameters(group_index)
                self._release_sub_group(group_index, timer_names)
            self._post_step(timer_names)

    DeepSpeedZeroOptimizer_Stage3.load_hp_checkpoint_state_from_checkpoint_dir_stage3 = (
        load_grouped_checkpoint
    )
    DeepSpeedZeroOptimizer_Stage3._physicalagent_grouped_universal_load = True
