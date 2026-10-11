#!/usr/bin/env python3
"""把清单里的行送进训练器自己的读取和 collate 代码，看训练时模型到底学什么。不加载模型权重，不需要 GPU。

    python3 training/tools/check_collate.py --model MODEL_DIR --manifest MANIFEST.json [--rows 20] [--max-input-tokens 196608]

走的是训练的同一条路：data_pipeline/manifest.py 读清单 -> training/jsonl_input.py 按字节位置取行 ->
training/vlm_collator.py 渲染、切出要算 loss 的部分。每行打印总长度、算 loss 的 token 数（思考 / 工具调用），
以及这两段开头的文字。任何一行被 collator 拒收都会报错退出。
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TextProcessor:
    """What the collator needs of a processor, for text-only rows: the tokenizer and the model's chat template."""

    def __init__(self, model):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        template = Path(model) / "chat_template.jinja"
        self.template = template.read_text(encoding="utf-8") if template.exists() else self.tokenizer.chat_template

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, chat_template=self.template, **kwargs)

    def __call__(self, text, images=None, **kwargs):
        if images:
            raise ValueError("text-only check: the row has images")
        return self.tokenizer(text, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="学生模型目录（tokenizer 和 chat_template.jinja）")
    parser.add_argument("--manifest", required=True, help="select_rows.py 写出的清单")
    parser.add_argument("--rows", type=int, default=20, help="随机检查多少行；0 = 全部")
    parser.add_argument("--max-input-tokens", type=int, default=196608)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from data_pipeline.manifest import load_manifest
    from training.jsonl_input import load_jsonl_input
    from training.vlm_collator import VlmCollator

    root, splits = load_manifest(args.manifest)
    dataset, estimates = load_jsonl_input(splits["train"])
    processor = TextProcessor(args.model)
    collate = VlmCollator(processor, root, args.max_input_tokens)
    order = list(range(len(dataset)))
    if args.rows and args.rows < len(order):
        order = sorted(random.Random(args.seed).sample(order, args.rows))
    print(f"{args.manifest}: {len(dataset)} rows, estimated tokens {estimates[0]}..{estimates[-1]}; collating {len(order)}")
    totals = [0, 0]
    for i in order:
        selected = splits["train"].rows[i]
        batch = collate([dataset[i]])
        ids, span = batch["input_ids"][0], batch["_sft_span"][0]
        think, call = ids[span == 1], ids[span == 2]
        totals[0] += len(think)
        totals[1] += len(call)
        show = lambda tokens: processor.tokenizer.decode(tokens[:40]).replace("\n", "\\n")[:110]  # noqa: E731
        print(f"{selected['trajectory_id'][-40:]} turn {selected['source_line']}: {len(ids)} tokens (manifest says {selected['estimated_tokens']}), "
              f"loss on {len(think)} reasoning + {len(call)} reply\n    reasoning: {show(think)}\n    reply:     {show(call)}")
        if len(ids) != selected["estimated_tokens"] and selected["estimated_tokens"] > 0:
            print("    note: the manifest's length is an estimate (lengths were not measured)" if abs(len(ids) - selected["estimated_tokens"]) > 0 else "")
    print(f"COLLATE_OK {len(order)} rows; loss tokens: {totals[0]} reasoning, {totals[1]} reply")


if __name__ == "__main__":
    main()
