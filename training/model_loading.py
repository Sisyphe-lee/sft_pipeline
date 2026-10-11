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
