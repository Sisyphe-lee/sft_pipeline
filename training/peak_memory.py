"""Per-rank GPU memory around each optimizer step, for sizing a run. Enabled with PRINT_PEAK_MEM=1.

`compute_loss` already prints the peak after the forward pass with the row's length; this adds what is held once the
model and optimizer are set up, and the peak of the whole step (forward, backward, update), then resets the peak so
that every step reports its own.
"""
import json

import torch
import torch.distributed as dist
from transformers import TrainerCallback


def _report(event, state):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({
        "event": event, "rank": rank, "global_step": int(state.global_step),
        "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 1),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1),
        "device_used_gb": round((total - free) / 2**30, 1),
    }, sort_keys=True), flush=True)


class PeakMemoryCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            _report("memory_after_setup", state)
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            _report("peak_memory_step", state)
            torch.cuda.reset_peak_memory_stats()
