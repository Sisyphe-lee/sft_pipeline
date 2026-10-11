"""Plan bounded SFT inputs without changing the immutable exported rows."""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta

from PIL import Image
import torch.distributed as dist

from physical_agent.artifact_refs import resolve_artifact_ref


def add_length_arguments(parser):
    parser.add_argument(
        "--max-input-tokens", type=int, default=73728,
        help="Maximum exact multimodal sequence length, including the completion.",
    )
    parser.add_argument(
        "--truncate-history", action=argparse.BooleanOptionalAction,
        default=True,
        help=("For over-limit rows only, remove oldest complete history rounds. "
              "With --no-truncate-history, filter over-limit rows before training."),
    )


def _image_count(message):
    content = message.get("content", [])
    if isinstance(content, str):
        return 0
    return sum(item.get("type") == "image" for item in content)


def history_boundaries(example):
    """Return user-message boundaries with no outstanding tool requests."""
    prompt = example["prompt"]
    validate_completion(example)
    users = [i for i, message in enumerate(prompt) if message.get("role") == "user"]
    if not users or users[-1] != len(prompt) - 1:
        raise ValueError("History truncation requires a final user observation")
    first = users[0]
    if any(m.get("role") not in {"system", "developer"} for m in prompt[:first]):
        raise ValueError("History must start after the system/developer prefix")
    if any(m.get("role") in {"system", "developer"} for m in prompt[first:]):
        raise ValueError("Cannot truncate interleaved system/developer instructions")
    pending = set()
    boundaries = []
    for i, message in enumerate(prompt[first:], first):
        role = message.get("role")
        if role == "user" and not pending:
            boundaries.append(i)
        elif role == "assistant":
            for call in message.get("tool_calls", []):
                call_id = call.get("id")
                if not call_id or call_id in pending:
                    raise ValueError("History has missing or duplicate tool call IDs")
                pending.add(call_id)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("History has an unmatched tool result")
            pending.remove(call_id)
    if pending or not boundaries or boundaries[-1] != users[-1]:
        raise ValueError("History has unresolved tool requests before the current observation")
    return boundaries


def validate_completion(example):
    completion = example.get("completion", [])
    if (not example.get("prompt") or len(completion) != 1
            or completion[0].get("role") != "assistant" or _image_count(completion[0])):
        raise ValueError("SFT requires a prompt and one text/tool assistant completion")


def crop_history(example, cut):
    """Remove whole messages and their image references; preserve the target."""
    if cut == 0:
        return example
    prompt = example["prompt"]
    first = next(i for i, m in enumerate(prompt) if m["role"] == "user")
    image_start = sum(_image_count(m) for m in prompt[:first])
    image_end = sum(_image_count(m) for m in prompt[:cut])
    result = dict(example)
    result["prompt"] = prompt[:first] + prompt[cut:]
    result["images"] = example["images"][:image_start] + example["images"][image_end:]
    if "image_roles" in example:
        result["image_roles"] = example["image_roles"][:image_start] + example["image_roles"][image_end:]
    return result


class RenderedLength:
    """Qwen processor length with image expansion, without decoding image pixels.

    Image headers determine the grids through the selected processor's own
    sizing rule. The collator independently verifies the actual tensor length.
    """

    def __init__(self, processor, dataset_root):
        self.processor = processor
        self.dataset_root = dataset_root
        self.sizes = {}

    def __call__(self, example, dataset_root=None):
        validate_completion(example)
        messages = example["prompt"] + example["completion"]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            tools=example.get("tools"),
        )
        ids = self.processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        sizes = []
        for ref in example["images"]:
            path = resolve_artifact_ref(ref, root=dataset_root or self.dataset_root)
            if path not in self.sizes:
                with Image.open(path) as image:
                    self.sizes[path] = (image.height, image.width)
            sizes.append(self.sizes[path])
        image_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if ids.count(image_id) != len(sizes):
            raise ValueError("Rendered image placeholders do not match image references")
        tokens = self.processor._get_num_multimodal_tokens(image_sizes=sizes).num_image_tokens
        return len(ids) + sum(n - 1 for n in tokens)


def _plan_length_policy_positions(dataset, processor, dataset_root, max_input_tokens,
                                  truncate_history, positions):
    """Measure a deterministic subset and return JSON-serializable decisions."""
    if max_input_tokens <= 0:
        raise ValueError("--max-input-tokens must be positive")
    measurer = RenderedLength(processor, dataset_root)
    entries = []
    for position in positions:
        position = int(position)
        record = dataset[position]
        example = json.loads(record["row_json"])

        def measure(row):
            return measurer(row, record.get("dataset_root", dataset_root))

        length = measure(example)
        if length > max_input_tokens:
            if not truncate_history:
                entries.append({"position": position, "status": "filtered", "tokens": int(length)})
                continue
            boundaries = history_boundaries(example)
            # The newest observation plus the fixed prefix and full completion
            # is the minimum protected input. Never truncate the target itself.
            if measure(crop_history(example, boundaries[-1])) > max_input_tokens:
                entries.append({"position": position, "status": "filtered", "tokens": int(length)})
                continue
            # Removing older complete rounds reduces length for the supported
            # Qwen templates. Find the largest retained suffix that fits.
            lo, hi = 1, len(boundaries) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if measure(crop_history(example, boundaries[mid])) <= max_input_tokens:
                    hi = mid
                else:
                    lo = mid + 1
            cut = boundaries[lo]
            entries.append({
                "position": position,
                "status": "truncated",
                "row_index": int(record["row_index"]),
                "cut": int(cut),
                # 裁剪后的精确长度:length_grouped 采样按它分组
                "tokens": int(measure(crop_history(example, cut))),
            })
        else:
            entries.append({"position": position, "status": "unchanged", "tokens": int(length)})
    return entries


def _merge_length_policy_entries(dataset, entries, max_input_tokens, truncate_history):
    """Validate and materialize independently measured row decisions."""
    ordered = sorted(entries, key=lambda entry: entry["position"])
    if [entry["position"] for entry in ordered] != list(range(len(dataset))):
        raise ValueError("Distributed length preflight returned missing or duplicate rows")
    keep, cuts, tokens = [], {}, []
    stats = dict(rows=len(dataset), unchanged=0, truncated=0, filtered=0,
                 max_input_tokens=max_input_tokens, truncate_history=truncate_history)
    for entry in ordered:
        status = entry["status"]
        if status not in {"unchanged", "truncated", "filtered"}:
            raise ValueError(f"Unknown length-policy status: {status}")
        stats[status] += 1
        if status == "filtered":
            continue
        keep.append(entry["position"])
        tokens.append(int(entry["tokens"]) if entry.get("tokens") is not None else -1)
        if status == "truncated":
            cuts[int(entry["row_index"])] = int(entry["cut"])
    stats["retained"] = len(keep)
    if not keep:
        raise ValueError("No SFT rows fit the length policy: " + json.dumps(stats, sort_keys=True))
    plan = {
        "version": 1,
        "keep_positions": keep,
        "history_cuts": {str(key): value for key, value in cuts.items()},
        # 与 keep_positions 对齐的精确渲染长度(含图片展开与裁剪后的 completion);
        # 只给 length_grouped 采样用,-1 表示旧缓存里没有。
        "tokens": tokens,
        "stats": stats,
    }
    return dataset.select(keep), cuts, stats, plan


def apply_length_policy_plan(dataset, plan, max_input_tokens, truncate_history):
    """Apply a validated cached plan without rendering/tokenizing rows again."""
    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise ValueError("Unsupported length-policy cache version")
    keep = plan.get("keep_positions")
    cuts = plan.get("history_cuts")
    stats = plan.get("stats")
    if not isinstance(keep, list) or not isinstance(cuts, dict) or not isinstance(stats, dict):
        raise ValueError("Malformed length-policy cache")
    if keep != sorted(set(keep)) or any(
        not isinstance(position, int) or position < 0 or position >= len(dataset)
        for position in keep
    ):
        raise ValueError("Length-policy cache contains invalid row positions")
    expected = {
        "rows": len(dataset),
        "retained": len(keep),
        "max_input_tokens": max_input_tokens,
        "truncate_history": truncate_history,
    }
    if any(stats.get(key) != value for key, value in expected.items()):
        raise ValueError("Length-policy cache does not match this dataset or configuration")
    selected_row_indices = {int(dataset[position]["row_index"]) for position in keep}
    parsed_cuts = {int(key): int(value) for key, value in cuts.items()}
    if not set(parsed_cuts).issubset(selected_row_indices):
        raise ValueError("Length-policy cache contains cuts for unselected rows")
    if any(cut <= 0 for cut in parsed_cuts.values()):
        raise ValueError("Length-policy cache contains invalid history cuts")
    if len(parsed_cuts) != stats.get("truncated", 0):
        raise ValueError("Length-policy cache truncated-row statistics are inconsistent")
    if stats.get("unchanged", 0) + stats.get("truncated", 0) != len(keep):
        raise ValueError("Length-policy cache retained-row statistics are inconsistent")
    if stats.get("filtered", 0) + len(keep) != len(dataset):
        raise ValueError("Length-policy cache filtered-row statistics are inconsistent")
    return dataset.select(keep), parsed_cuts, dict(stats)


def prepare_length_policy(dataset, processor, dataset_root, max_input_tokens,
                          truncate_history=True):
    """Select rows and plan cuts on one process."""
    entries = _plan_length_policy_positions(
        dataset, processor, dataset_root, max_input_tokens, truncate_history,
        range(len(dataset)),
    )
    selected, cuts, stats, _ = _merge_length_policy_entries(
        dataset, entries, max_input_tokens, truncate_history,
    )
    return selected, cuts, stats


def prepare_length_policy_distributed(dataset, processor, dataset_root, max_input_tokens,
                                      truncate_history=True):
    """Shard exact preflight across torchrun ranks and reconstruct one shared plan."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size == 1:
        entries = _plan_length_policy_positions(
            dataset, processor, dataset_root, max_input_tokens, truncate_history,
            range(len(dataset)),
        )
        return _merge_length_policy_entries(
            dataset, entries, max_input_tokens, truncate_history,
        )
    if not dist.is_available() or dist.is_initialized():
        raise RuntimeError("Length preflight requires a fresh torch.distributed process group")

    dist.init_process_group("gloo", timeout=timedelta(minutes=30))
    try:
        try:
            entries = _plan_length_policy_positions(
                dataset, processor, dataset_root, max_input_tokens, truncate_history,
                range(rank, len(dataset), world_size),
            )
            payload = {"rank": rank, "entries": entries, "error": None}
        except Exception as exc:  # Propagate one-rank data errors without hanging peers.
            payload = {
                "rank": rank,
                "entries": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload)
        failures = [item for item in gathered if item["error"]]
        if failures:
            first = min(failures, key=lambda item: item["rank"])
            raise ValueError(
                f"Length preflight failed on rank {first['rank']}: {first['error']}"
            )
        merged = [entry for item in gathered for entry in item["entries"]]
        return _merge_length_policy_entries(
            dataset, merged, max_input_tokens, truncate_history,
        )
    finally:
        dist.destroy_process_group()
