#!/usr/bin/env python3
"""合并门辅助:同一行按两条读数路径(老 CLI 的 json 往返 / 新 CLI 的按 offset 读原始字节)喂给
合并后的 VlmCollator,比 input_ids / labels 是否逐位相同。相同 → loss 差异不来自输入渲染。

用法:PYTHONPATH=<merge repo>:<qwendeps...> python3 training/tools/check_render_parity.py --model <ckpt> --jsonl <rows.jsonl> [--rows 3]
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from training.vlm_collator import VlmCollator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--rows", type=int, default=3)
    a = ap.parse_args()
    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    for name in ("pad_token", "eos_token", "pad_token_id", "eos_token_id"):
        if not hasattr(processor, name):
            setattr(processor, name, getattr(tokenizer, name))
    if not hasattr(processor, "convert_tokens_to_ids"):
        processor.convert_tokens_to_ids = tokenizer.convert_tokens_to_ids
    collator = VlmCollator(processor, a.jsonl.parent, 10**9)
    raw_lines = a.jsonl.read_bytes().split(b"\n")
    bad = 0
    for i in range(a.rows):
        raw = raw_lines[i].decode("utf-8")                      # 新 CLI:原始字节
        roundtrip = json.dumps(json.loads(raw), ensure_ascii=False)  # 老 CLI:loads → dumps
        outs = []
        for text in (raw, roundtrip):
            b = collator([{"row_json": text, "row_index": i}])
            ids = b["input_ids"][0].tolist(); lab = b["labels"][0].tolist()
            outs.append((hashlib.sha256(json.dumps([ids, lab]).encode()).hexdigest()[:12], len(ids), sum(1 for x in lab if x != -100),
                         b.get("pixel_values").shape if b.get("pixel_values") is not None else None))
        same = outs[0][0] == outs[1][0]
        bad += (not same)
        print(f"row {i}: raw={outs[0]} roundtrip={outs[1]} same={same} text_equal={raw == roundtrip}")
    print("RENDER_PARITY", "OK" if bad == 0 else f"MISMATCH rows={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
