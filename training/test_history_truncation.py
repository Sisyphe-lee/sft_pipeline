"""CPU regression tests using the real Qwen multimodal processor."""

import argparse
import copy
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path

import torch
import torch.multiprocessing as mp
from datasets import Dataset
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor,
)

from training.history_truncation import (
    RenderedLength, _merge_length_policy_entries, _plan_length_policy_positions,
    add_length_arguments, apply_length_policy_plan, crop_history, history_boundaries,
    prepare_length_policy, prepare_length_policy_distributed,
)
from training.vlm_collator import VlmCollator
from training.jsonl_input import load_jsonl_input


TEMPLATE = """{% if tools %}{{ tools | tojson }}{% endif %}{% for m in messages %}<|im_start|>{{ m.role }}
{% if m.role == 'assistant' %}<think>{{ m.get('reasoning_content', '') }}</think>{% endif %}{% if m.content is string %}{{ m.content }}{% else %}{% for c in m.content %}{% if c.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ c.text }}{% endif %}{% endfor %}{% endif %}{% if m.tool_calls %}<tool_call>{{ m.tool_calls | tojson }}</tool_call>{% endif %}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
<think>{% endif %}"""


def make_processor():
    special = ["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
               "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<think>",
               "</think>", "<tool_call>", "</tool_call>"]
    words = ["system", "user", "assistant", "tool", "protocol", "old", "middle",
             "current", "history_reasoning", "result", "target_reasoning", "answer"]
    backend = Tokenizer(models.WordLevel({s: i for i, s in enumerate(special + words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        eos_token="<|im_end|>", additional_special_tokens=special[2:],
    )
    return Qwen3VLProcessor(
        tokenizer=tokenizer, chat_template=TEMPLATE,
        video_processor=Qwen3VLVideoProcessor(),
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, merge_size=2, min_pixels=1024, max_pixels=1048576,
        ),
    )


def call(call_id):
    return {"id": call_id, "type": "function", "function": {"name": "move", "arguments": {"x": 1}}}


def _distributed_length_rank(rank, port, root, records, limit):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
    )
    dataset = Dataset.from_list(records)
    selected, cuts, stats, plan = prepare_length_policy_distributed(
        dataset, make_processor(), Path(root), limit,
    )
    assert not torch.distributed.is_initialized()
    torch.distributed.init_process_group("gloo")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    Path(root, f"rank-{rank}.json").write_text(json.dumps({
        "rows": list(selected["row_index"]), "cuts": cuts, "stats": stats, "plan": plan,
    }, sort_keys=True))


class HistoryTruncationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = make_processor()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name, size, color in [("old", (64, 64), "red"), ("middle", (96, 64), "green"), ("current", (128, 96), "blue")]:
            Image.new("RGB", size, color).save(self.root / (name + ".png"))
        def observation(word):
            return {"role": "user", "content": [{"type": "text", "text": (word + " ") * 60}, {"type": "image"}]}
        self.row = {
            "prompt": [
                {"role": "system", "content": "protocol"},
                observation("old"),
                {"role": "assistant", "content": "history_reasoning", "tool_calls": [call("a"), call("b")]},
                {"role": "tool", "content": "result", "tool_call_id": "a"},
                {"role": "tool", "content": "result", "tool_call_id": "b"},
                observation("middle"),
                {"role": "assistant", "content": "history_reasoning", "tool_calls": [call("c")]},
                {"role": "tool", "content": "result", "tool_call_id": "c"},
                observation("current"),
            ],
            "completion": [{"role": "assistant", "reasoning_content": "target_reasoning",
                            "content": "answer", "tool_calls": [call("target")]}],
            "tools": [{"type": "function", "function": {"name": "move"}}],
            "images": ["old.png", "middle.png", "current.png"],
            "image_roles": ["old", "middle", "current"],
        }
        self.measure = RenderedLength(self.processor, self.root)

    def dataset(self, rows):
        return Dataset.from_list([{"row_index": 10 + i, "row_json": json.dumps(r)} for i, r in enumerate(rows)])

    def collate(self, dataset, limit, cuts=None):
        return VlmCollator(self.processor, self.root, limit, history_cuts=cuts or {})([dataset[0]])

    def test_cli_default_and_opt_out(self):
        parser = argparse.ArgumentParser()
        add_length_arguments(parser)
        self.assertTrue(parser.parse_args([]).truncate_history)
        self.assertEqual(parser.parse_args([]).max_input_tokens, 73728)
        self.assertFalse(parser.parse_args(["--no-truncate-history"]).truncate_history)

    def test_under_limit_and_exact_limit_are_unchanged(self):
        original = copy.deepcopy(self.row)
        dataset = self.dataset([self.row])
        limit = self.measure(self.row)
        outputs = []
        for enabled in [True, False]:
            selected, cuts, stats = prepare_length_policy(dataset, self.processor, self.root, limit, enabled)
            self.assertEqual(cuts, {})
            self.assertEqual(stats["unchanged"], 1)
            self.assertEqual(selected[0], dataset[0])
            outputs.append(self.collate(selected, limit, cuts))
        self.assertEqual(self.row, original)
        self.assertIs(crop_history(self.row, 0), self.row)
        self.assertEqual(outputs[0]["input_ids"].shape[-1], limit)
        for key in outputs[0]:
            self.assertTrue(torch.equal(outputs[0][key], outputs[1][key]), key)

    def test_whole_rounds_images_and_target_mask(self):
        self.assertEqual(history_boundaries(self.row), [1, 5, 8])
        expected = crop_history(self.row, 5)
        limit = self.measure(expected)
        dataset = self.dataset([self.row])
        selected, cuts, stats = prepare_length_policy(dataset, self.processor, self.root, limit)
        self.assertEqual(cuts, {10: 5})
        self.assertEqual(stats["truncated"], 1)
        self.assertEqual(expected["images"], ["middle.png", "current.png"])
        self.assertEqual(expected["image_roles"], ["middle", "current"])
        self.assertEqual(expected["prompt"], self.row["prompt"][:1] + self.row["prompt"][5:])
        self.assertIs(expected["completion"], self.row["completion"])
        self.assertIs(expected["tools"], self.row["tools"])
        original = self.collate(dataset, self.measure(self.row))
        cropped = self.collate(selected, limit, cuts)
        self.assertEqual(cropped["input_ids"].shape[-1], limit)
        self.assertEqual(cropped["image_grid_thw"].shape[0], 2)
        self.assertTrue(torch.equal(original["image_grid_thw"][1:], cropped["image_grid_thw"]))
        old_patches = int(original["image_grid_thw"][0].prod())
        self.assertTrue(torch.equal(original["pixel_values"][old_patches:], cropped["pixel_values"]))
        self.assertTrue(torch.equal(original["labels"][original["labels"] != -100],
                                    cropped["labels"][cropped["labels"] != -100]))
        labels = cropped["labels"][0]
        start = int((labels != -100).nonzero()[0])
        self.assertTrue((labels[:start] == -100).all())
        text = self.processor.tokenizer.decode(labels[start:])
        self.assertIn("target_reasoning", text)
        self.assertIn("answer", text)
        self.assertIn("<tool_call>", text)
        self.assertIn("<|im_end|>", text)
        self.assertNotIn("history_reasoning", text)
        self.assertNotIn("result", text)

    def test_disabled_and_irreducible_rows_are_filtered_before_loader(self):
        minimum = crop_history(self.row, 8)
        huge = copy.deepcopy(minimum)
        huge["completion"][0]["reasoning_content"] = "target_reasoning " * 2000
        dataset = self.dataset([huge, self.row, minimum])
        limit = self.measure(minimum)
        for enabled, expected_indices in [(True, [11, 12]), (False, [12])]:
            selected, cuts, stats = prepare_length_policy(dataset, self.processor, self.root, limit, enabled)
            self.assertEqual(selected["row_index"], expected_indices)
            self.assertEqual(stats["filtered"], 1 if enabled else 2)
            collator = VlmCollator(self.processor, self.root, limit, history_cuts=cuts)
            zero = list(DataLoader(selected, batch_size=1, num_workers=0, collate_fn=collator))
            workers = list(DataLoader(selected, batch_size=1, num_workers=2, collate_fn=collator))
            self.assertEqual(len(workers), len(expected_indices))
            for a, b in zip(zero, workers):
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_sharded_decisions_and_cached_plan_match_single_process(self):
        minimum = crop_history(self.row, 8)
        huge = copy.deepcopy(minimum)
        huge["completion"][0]["reasoning_content"] = "target_reasoning " * 2000
        dataset = self.dataset([huge, self.row, minimum, self.row])
        limit = self.measure(minimum)
        expected, expected_cuts, expected_stats = prepare_length_policy(
            dataset, self.processor, self.root, limit,
        )
        shards = [
            _plan_length_policy_positions(
                dataset, self.processor, self.root, limit, True,
                range(rank, len(dataset), 3),
            )
            for rank in range(3)
        ]
        selected, cuts, stats, plan = _merge_length_policy_entries(
            dataset, [entry for shard in shards for entry in shard], limit, True,
        )
        cached, cached_cuts, cached_stats = apply_length_policy_plan(
            dataset, json.loads(json.dumps(plan)), limit, True,
        )
        self.assertEqual(selected["row_index"], expected["row_index"])
        self.assertEqual(cached["row_index"], expected["row_index"])
        self.assertEqual(cuts, expected_cuts)
        self.assertEqual(cached_cuts, expected_cuts)
        self.assertEqual(stats, expected_stats)
        self.assertEqual(cached_stats, expected_stats)
        broken = copy.deepcopy(plan)
        broken["keep_positions"].append(broken["keep_positions"][-1])
        with self.assertRaisesRegex(ValueError, "invalid row positions"):
            apply_length_policy_plan(dataset, broken, limit, True)

    def test_two_rank_gloo_preflight_matches_single_process(self):
        minimum = crop_history(self.row, 8)
        records = [
            {"row_index": 10 + i, "row_json": json.dumps(row)}
            for i, row in enumerate([self.row, minimum, self.row, minimum])
        ]
        dataset = Dataset.from_list(records)
        limit = self.measure(minimum)
        expected, cuts, stats = prepare_length_policy(
            dataset, self.processor, self.root, limit,
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(
            _distributed_length_rank,
            args=(port, str(self.root), records, limit),
            nprocs=2,
            join=True,
        )
        outputs = [json.loads((self.root / f"rank-{rank}.json").read_text()) for rank in range(2)]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0]["rows"], expected["row_index"])
        self.assertEqual({int(key): value for key, value in outputs[0]["cuts"].items()}, cuts)
        self.assertEqual(outputs[0]["stats"], stats)

    def test_invalid_structure_and_empty_selection_fail_before_training(self):
        invalid = copy.deepcopy(self.row)
        invalid["prompt"][4]["tool_call_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unmatched"):
            history_boundaries(invalid)
        invalid = copy.deepcopy(self.row)
        invalid["completion"].append({"role": "tool", "content": "result"})
        with self.assertRaisesRegex(ValueError, "assistant completion"):
            history_boundaries(invalid)
        with self.assertRaisesRegex(ValueError, "No SFT rows"):
            prepare_length_policy(self.dataset([self.row]), self.processor, self.root, 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            prepare_length_policy(self.dataset([self.row]), self.processor, self.root, 0)
        invalid = copy.deepcopy(self.row)
        invalid["images"].pop()
        with self.assertRaisesRegex(ValueError, "placeholders"):
            self.measure(invalid)

    def test_lazy_index_selection_keeps_source_indices(self):
        from data_pipeline.select_sft_rows import main as select_main
        from data_pipeline.manifest import load_manifest
        minimum = crop_history(self.row, 8)
        huge = copy.deepcopy(minimum)
        huge["completion"][0]["reasoning_content"] = "target_reasoning " * 2000
        path = self.root / "input.jsonl"
        examples = [huge, self.row, minimum]
        for i, example in enumerate(examples):
            example["metadata"] = {"context_manifest": {"estimated_tokens": 99999},
                                   "task": {"id": "cup", "seed": 1},
                                   "trajectory_id": "one", "turn": i}
        content = "\n".join(json.dumps(r) for r in examples)
        path.write_text(content)
        policy = self.root / 'policy.yaml'
        policy.write_text('official_success_only: false\n')
        manifest = self.root / 'train.json'
        select_main(['--dataset', str(path), '--policy', str(policy), '--output', str(manifest)])
        dataset, _ = load_jsonl_input(load_manifest(manifest)[1]['train'])
        limit = self.measure(minimum)
        selected, cuts, _ = prepare_length_policy(dataset, self.processor, self.root, limit)
        self.assertEqual([r["row_index"] for r in selected], [1, 2])
        self.assertEqual(cuts, {1: 8})
        self.assertEqual(selected[0]["row_json"], dataset[1]["row_json"])
        self.assertEqual(path.read_text(), content)
        collator = VlmCollator(self.processor, self.root, limit, history_cuts=cuts)
        batches = list(DataLoader(selected, batch_size=1, num_workers=2, collate_fn=collator))
        self.assertEqual([b["_sft_row_index"].item() for b in batches], [1, 2])
        self.assertTrue(all(b["input_ids"].shape[-1] <= limit for b in batches))

    def test_truncation_can_remove_last_remaining_old_image(self):
        row = copy.deepcopy(self.row)
        for message in row["prompt"][5:]:
            if isinstance(message["content"], list):
                message["content"] = [c for c in message["content"] if c["type"] != "image"]
        row["images"] = row["images"][:1]
        row["image_roles"] = row["image_roles"][:1]
        minimum = crop_history(row, 8)
        limit = self.measure(minimum)
        selected, cuts, _ = prepare_length_policy(self.dataset([row]), self.processor, self.root, limit)
        batch = self.collate(selected, limit, cuts)
        self.assertNotIn("pixel_values", batch)
        self.assertEqual(batch["input_ids"].shape[-1], limit)


if __name__ == "__main__":
    unittest.main()
