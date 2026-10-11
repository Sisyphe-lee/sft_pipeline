"""Multimodal collation with completion masks and preplanned history cuts."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import torch
from PIL import Image

from physical_agent.artifact_refs import resolve_artifact_ref
from training.history_truncation import crop_history, validate_completion


@dataclass
class VlmCollator:
    processor: object
    dataset_root: Path
    max_input_tokens: int
    history_cuts: dict[int, int] = field(default_factory=dict)

    def __call__(self, examples):
        if len(examples) != 1:
            raise ValueError("Qwen3-VL SFT currently requires per-device batch size 1")
        example = json.loads(examples[0]["row_json"])
        validate_completion(example)
        example = crop_history(example, self.history_cuts.get(int(examples[0]["row_index"]), 0))
        dataset_root = examples[0].get("dataset_root", self.dataset_root)
        images = [
            Image.open(resolve_artifact_ref(path, root=dataset_root)).convert("RGB")
            for path in example["images"]
        ]
        if "prompt" in example and "completion" in example:
            full_text = self.processor.apply_chat_template(
                example["prompt"] + example["completion"],
                tokenize=False,
                add_generation_prompt=False,
                tools=example.get("tools"),
            )
            prompt_text = self.processor.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=True,
                tools=example.get("tools"),
            )
            inputs = self.processor(
                text=[full_text], images=images or None, return_tensors="pt", padding=True
            )
            prompt_inputs = self.processor(
                text=[prompt_text], images=images or None, return_tensors="pt", padding=True
            )
            prompt_ids = prompt_inputs["input_ids"][0]
            full_ids = inputs["input_ids"][0]
            prompt_length = prompt_ids.numel()
            if (
                full_ids.numel() > self.max_input_tokens
            ):
                raise ValueError(
                    "rendered SFT row exceeds the configured exact token limit: "
                    f"row_index={examples[0]['row_index']}, "
                    f"tokens={full_ids.numel()}, limit={self.max_input_tokens}. "
                    "Length preflight and actual processor output disagree."
                )
            if prompt_length > full_ids.numel() or not full_ids[:prompt_length].equal(
                prompt_ids
            ):
                raise ValueError(
                    "The rendered prompt is not an exact token prefix of the full "
                    "conversation; refusing to train with ambiguous completion labels."
                )
        if inputs["input_ids"].shape[-1] > self.max_input_tokens:
            raise ValueError("Actual processor output exceeds the preflight token limit")
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100
        ignored_ids = {
            self.processor.tokenizer.pad_token_id,
            self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            self.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        }
        for token_id in ignored_ids:
            if token_id is not None:
                labels[labels == token_id] = -100
        # 监督段再切成两半,给 W&B 分开看 accuracy:
        #   1 = reasoning —— 从 <think> 之后到 </think>(含)为止;
        #   2 = tool_call —— </think> 之后的一切(换行、<tool_call>…</tool_call>、<|im_end|>)。
        # 没有 </think> 的行整段算 2。按 label 所在位置标,损失里的 shift 由 trainer 那边做。
        # 历史 turn 里也有 </think>,但它们在 prompt 段、已经是 -100,不会被算进来。
        span = completion_spans(labels, inputs["input_ids"], self.processor.tokenizer)
        inputs["labels"] = labels
        inputs["_sft_span"] = span
        inputs["_sft_row_index"] = torch.tensor(
            [int(examples[0]["row_index"])], dtype=torch.int64
        )
        return inputs


def completion_spans(labels, input_ids, tokenizer):
    """labels 同形状的 int8:0 = 不监督,1 = reasoning,2 = tool_call。见 VlmCollator 里的注释。"""
    span = torch.zeros_like(labels, dtype=torch.int8)
    supervised = labels != -100
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    positions = torch.arange(labels.shape[-1], device=labels.device).expand_as(labels)
    for row in range(labels.shape[0]):
        hits = ((input_ids[row] == think_end) & supervised[row]).nonzero()
        if hits.numel():
            cut = int(hits[0])
            span[row][supervised[row] & (positions[row] <= cut)] = 1
            span[row][supervised[row] & (positions[row] > cut)] = 2
        else:
            span[row][supervised[row]] = 2
    return span
