"""Small helpers for opt-in ZeRO-3 initialization of large checkpoints."""

import json
import os
import time
from pathlib import Path

import torch


def validate_zero3_init(config):
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    if not isinstance(config, dict) or config.get("zero_optimization", {}).get("stage") != 3:
        raise ValueError("--zero3-init requires a DeepSpeed ZeRO stage 3 configuration")


def parameter_numel(parameter):
    """Report global size even while a ZeRO parameter's local storage is empty."""
    size = getattr(parameter, "ds_numel", None)
    return parameter.numel() if size is None else size



def _wait_for(paths, seconds=1800):
    deadline = time.monotonic() + seconds
    while not all(os.path.exists(path) for path in paths):
        if time.monotonic() > deadline:
            raise TimeoutError(f"waited {seconds}s for {[p for p in paths if not os.path.exists(p)]}")
        time.sleep(1)


def hold_whole(model, fragments, shared_dir="/dev/shm"):
    """Take the parameters whose names contain a fragment out of ZeRO-3: gather each once and keep it whole and
    frozen in host memory, as a buffer of its module under the same name. The ranks of one node share one copy
    (a file in `shared_dir` that local rank 0 writes and every local rank maps; it is unlinked once all have).
    Returns the number of elements held.

    For FlashNext's n-gram embedding, one tensor of 51.2B elements, 95 GiB in bf16. The model looks its rows up
    on whatever device the table is on, so the table can stay off the GPUs. Measured on 8 B300 on 2026-10-12: as a
    ZeRO-3 parameter it was gathered onto every GPU at every step, one copy and two while assembling, whatever the
    number of GPUs, which is what filled the cards of the 32-GPU jobs; kept on every GPU it left too little room
    for the optimizer state of full fine-tuning beside about 90 GiB of activations at 50k tokens."""
    held, local_rank = 0, int(os.environ.get("LOCAL_RANK", "0"))
    others = [rank for rank in range(int(os.environ.get("LOCAL_WORLD_SIZE", "1"))) if rank != local_rank]
    for module_name, module in model.named_modules():
        for name, parameter in list(module._parameters.items()):
            full_name = f"{module_name}.{name}" if module_name else name
            if parameter is None or not any(fragment in full_name for fragment in fragments):
                continue
            if hasattr(parameter, "all_gather"):  # a ZeRO-3 parameter: every rank gathers it here, together
                parameter.all_gather()
            source = parameter.data
            shape, dtype, count = source.shape, source.dtype, source.numel()
            path = os.path.join(shared_dir, f"sft_held_{os.getppid()}_{full_name}.bin")  # the launcher's pid: one per node and run
            if local_rank == 0:
                shared = torch.from_file(path + ".tmp", shared=True, size=count, dtype=dtype)
                flat = source.reshape(-1)
                for begin in range(0, count, 2**30):
                    shared[begin:begin + 2**30].copy_(flat[begin:begin + 2**30])
                del flat
                os.replace(path + ".tmp", path)
            del module._parameters[name], source
            parameter.data = torch.empty(0, dtype=dtype)  # frees this rank's whole copy
            if hasattr(parameter, "ds_tensor"):
                parameter.ds_tensor = None  # and its partition
            if local_rank == 0:
                marks = [f"{path}.mapped{rank}" for rank in others]
                _wait_for(marks)
                for used in marks + [path]:
                    os.unlink(used)
            else:
                _wait_for([path])
                shared = torch.from_file(path, shared=True, size=count, dtype=dtype)
                open(f"{path}.mapped{local_rank}", "w").close()
            module.register_buffer(name, shared.view(shape))
            held += count
    return held
