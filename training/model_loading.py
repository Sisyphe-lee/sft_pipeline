"""Small helpers for opt-in ZeRO-3 initialization of large checkpoints."""

import json
from pathlib import Path


def validate_zero3_init(config):
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    if not isinstance(config, dict) or config.get("zero_optimization", {}).get("stage") != 3:
        raise ValueError("--zero3-init requires a DeepSpeed ZeRO stage 3 configuration")


def parameter_numel(parameter):
    """Report global size even while a ZeRO parameter's local storage is empty."""
    size = getattr(parameter, "ds_numel", None)
    return parameter.numel() if size is None else size


def hold_whole(model, fragments):
    """Take the parameters whose names contain a fragment out of ZeRO-3: gather each once and keep it whole and
    frozen on every GPU, as a buffer of its module under the same name. Returns the number of elements held.

    ZeRO-3 gathers a partitioned parameter onto every GPU each time its module runs, and assembling it holds two
    copies. FlashNext's n-gram embedding is one tensor of 51.2B elements, 95 GiB in bf16: measured on 2026-10-12,
    its gather was 95 to 190 GiB on every GPU at every step, whatever the number of GPUs, which is what filled the
    cards of the 32-GPU jobs. A buffer is gathered once, before the optimizer state exists, is saved under the same
    key, and is never updated."""
    held = 0
    for module_name, module in model.named_modules():
        for name, parameter in list(module._parameters.items()):
            full_name = f"{module_name}.{name}" if module_name else name
            if parameter is None or not any(fragment in full_name for fragment in fragments):
                continue
            if hasattr(parameter, "all_gather"):  # a ZeRO-3 parameter: every rank gathers it here, together
                parameter.all_gather()
            data = parameter.data
            del module._parameters[name]
            module.register_buffer(name, data)
            if hasattr(parameter, "ds_tensor"):
                parameter.ds_tensor = None  # this rank's partition, no longer needed
            held += data.numel()
    return held
