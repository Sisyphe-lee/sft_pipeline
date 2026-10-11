#!/usr/bin/env python3
"""按 train_sft.py 的渲染方式精确统计每行 SFT 的 token 长度。

train_sft.py 对超过 ``--max-input-tokens`` 的行会在长度 preflight 里标成 filtered
直接跳过(``truncate_history: false`` 时;见 training/history_truncation.py),不报错。
所以换数据前必须先知道有多少行会撞上限 —— 否则训练会安静地少训一批行,
而 manifest 里记的行数还是原来那个。这里复刻它的渲染:
``processor.apply_chat_template(prompt + completion, tools=...)``,文本部分
用 tokenizer 精确计数;图片部分 Qwen3.5 processor 会把每个 image 占位展开成
固定数量的 token(同一分辨率恒定),用一张真图跑一次 processor 拿到展开数,
再按张数乘回去 —— 与逐行喂图的结果逐位一致,但 2000 行几分钟就能跑完。

用法:
  python3 measure_rendered_tokens.py --model <processor 目录> --dataset <jsonl> \
      --limit 73728 [--report out.json] [--max-rows N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physical_agent.artifact_refs import resolve_artifact_ref  # noqa: E402


def quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    xs = sorted(values)
    n = len(xs)
    return {
        "min": xs[0],
        "p50": xs[n // 2],
        "p90": xs[int(0.9 * (n - 1))],
        "p99": xs[int(0.99 * (n - 1))],
        "max": xs[-1],
    }


def summarize(rows_out: list[dict], dataset: str, limit: int, per_image: dict) -> dict:
    per_task_total: Counter = Counter()
    per_task_over: Counter = Counter()
    for r in rows_out:
        per_task_total[r["task"]] += 1
        if r["tokens_full"] > limit:
            per_task_over[r["task"]] += 1
    return {
        "dataset": dataset, "rows": len(rows_out), "limit": limit,
        "rows_over_limit": sum(per_task_over.values()), "per_image_expansion": per_image,
        "tokens_full": quantiles([r["tokens_full"] for r in rows_out]),
        "completion_tokens": quantiles([r["completion_tokens"] for r in rows_out]),
        "per_task": {t: {"rows": per_task_total[t], "over": per_task_over[t]} for t in sorted(per_task_total)},
    }


def merge_reports(paths: list[str], out: Path, limit: int) -> int:
    rows: dict[int, dict] = {}
    per_image: dict = {}
    dataset = None
    for p in paths:
        d = json.load(open(p))
        dataset = dataset or d["summary"]["dataset"]
        per_image.update(d["summary"].get("per_image_expansion", {}))
        for r in d["rows"]:
            rows[r["row_index"]] = r
    rows_out = [rows[i] for i in sorted(rows)]
    if rows_out and sorted(rows) != list(range(rows_out[0]["row_index"], rows_out[-1]["row_index"] + 1)):
        raise SystemExit("分片合并后 row_index 不连续,有分片缺失")
    summary = summarize(rows_out, dataset, limit, per_image)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    out.write_text(json.dumps({"summary": summary, "rows": rows_out}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="含 processor/tokenizer 的目录")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=73728)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--images-root", help="按这个目录解析相对图片路径(默认 jsonl 所在目录;物化没拷完时指到源数据的同层目录)")
    ap.add_argument("--start-row", type=int, default=0, help="分片:从这一行(含)开始")
    ap.add_argument("--end-row", type=int, help="分片:到这一行(不含)结束")
    ap.add_argument("--merge", nargs="*", help="只合并这些分片 report(按 row_index 排序去重),写到 --report")
    args = ap.parse_args()
    if args.merge:
        return merge_reports(args.merge, args.report, args.limit)

    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    dataset_root = Path(args.images_root).resolve() if args.images_root else args.dataset.resolve().parent

    # 用第一行的第一张真图标定"一张图展开成多少个 token"。
    per_image: dict[tuple[int, int], int] = {}

    def image_expansion(path: str) -> int:
        img = Image.open(resolve_artifact_ref(path, root=dataset_root)).convert("RGB")
        key = img.size
        if key not in per_image:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            with_img = processor(text=[text], images=[img], return_tensors="pt")["input_ids"][0].numel()
            without = len(tokenizer(text)["input_ids"])
            per_image[key] = with_img - without  # 展开后比占位多出来的 token 数
        return per_image[key]

    rows_out = []
    over = []
    per_task_total: Counter = Counter()
    per_task_over: Counter = Counter()
    with args.dataset.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx < args.start_row:
                continue
            if args.end_row is not None and idx >= args.end_row:
                break
            if args.max_rows is not None and idx >= args.start_row + args.max_rows:
                break
            row = json.loads(line)
            full_text = processor.apply_chat_template(
                row["prompt"] + row["completion"], tokenize=False,
                add_generation_prompt=False, tools=row.get("tools"),
            )
            prompt_text = processor.apply_chat_template(
                row["prompt"], tokenize=False, add_generation_prompt=True, tools=row.get("tools"),
            )
            n_text_full = len(tokenizer(full_text)["input_ids"])
            n_text_prompt = len(tokenizer(prompt_text)["input_ids"])
            extra = sum(image_expansion(p) for p in row["images"])
            tokens_full = n_text_full + extra
            tokens_prompt = n_text_prompt + extra
            meta = row.get("metadata", {})
            task = (meta.get("task") or {}).get("id") or meta.get("task_id") or "?"
            rec = {
                "row_index": idx, "task": task, "turn": meta.get("turn"),
                "n_images": len(row["images"]), "tokens_full": tokens_full,
                "tokens_prompt": tokens_prompt, "completion_tokens": tokens_full - tokens_prompt,
            }
            rows_out.append(rec)
            per_task_total[task] += 1
            if tokens_full > args.limit:
                over.append(rec)
                per_task_over[task] += 1
            if idx % 200 == 0:
                print(f"row {idx} tokens_full={tokens_full}", flush=True)

    summary = summarize(rows_out, str(args.dataset), args.limit,
                        {f"{k[0]}x{k[1]}": v for k, v in per_image.items()})
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if args.report:
        args.report.write_text(json.dumps({"summary": summary, "rows": rows_out}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
