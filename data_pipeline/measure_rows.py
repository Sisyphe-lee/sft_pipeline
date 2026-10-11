#!/usr/bin/env python3
"""量长度：用学生模型的对话模板和 tokenizer 把每一行按训练器的方式渲染一遍，记下 token 数和训练器会拒收的行。

    python3 data_pipeline/measure_rows.py --model MODEL_DIR --rows DATA_DIR/rows [--out DATA_DIR/tokens] [--workers 32]

写出 DATA_DIR/tokens/<rollout>.jsonl，和训练行一一对应，每行：

    {"turn": N, "tokens": 整行 token 数, "prompt_tokens": 输入部分, "ok": true|false, "problem": null 或原因}

训练器（training/vlm_collator.py）对每一行渲染两次：输入 + 回复，以及只有输入并加上生成提示；输入的 token 必须是
整行 token 的前缀，否则报错退出。这里做同样的两次渲染和同样的检查，所以 ok 的行训练器一定收。选数据时用 --tokens
读这些文件：不 ok 或超长的行不选。已量过且行数相符的文件跳过。不需要 GPU，只读 MODEL_DIR 里的 tokenizer 和模板。
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

STATE = {}


def start(model):
    from transformers import AutoTokenizer

    STATE["tokenizer"] = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    template = Path(model) / "chat_template.jinja"
    STATE["template"] = template.read_text(encoding="utf-8") if template.exists() else STATE["tokenizer"].chat_template


def measure(job):
    path, offset, length = job
    with open(path, "rb") as stream:
        stream.seek(offset)
        row = json.loads(stream.read(length))
    tokenizer, template = STATE["tokenizer"], STATE["template"]
    out = dict(turn=row["metadata"]["turn"], tokens=None, prompt_tokens=None, ok=False, problem=None)
    try:
        render = lambda messages, more: tokenizer.apply_chat_template(  # noqa: E731
            messages, tools=row.get("tools") or None, chat_template=template, tokenize=False, add_generation_prompt=more)
        full = tokenizer(render(row["prompt"] + row["completion"], False))["input_ids"]
        prompt = tokenizer(render(row["prompt"], True))["input_ids"]
    except Exception as error:  # noqa: BLE001 - a template error is a finding about the row, not a crash
        out["problem"] = f"template: {type(error).__name__}: {str(error)[:120]}"
        return out
    out.update(tokens=len(full), prompt_tokens=len(prompt))
    if len(prompt) >= len(full) or full[:len(prompt)] != prompt:
        out["problem"] = "the input is not a token prefix of the whole row"
    else:
        out["ok"] = True
    return out


def offsets(path):
    jobs, offset = [], 0
    with open(path, "rb") as stream:
        for line in stream:
            jobs.append((str(path), offset, len(line)))
            offset += len(line)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="学生模型目录（要有 tokenizer 和 chat_template.jinja）")
    parser.add_argument("--rows", required=True, type=Path, help="annotate.py 写出的训练行目录")
    parser.add_argument("--out", type=Path, help="默认是 rows 旁边的 tokens/")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--only", help="只量文件名里含这段文字的 rollout")
    args = parser.parse_args()
    out = args.out or args.rows.parent / "tokens"
    out.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=start, initargs=(args.model,)) as pool:
        for path in sorted(args.rows.glob("*.jsonl")):
            if args.only and args.only not in path.name:
                continue
            jobs, target = offsets(path), out / path.name
            if target.exists() and sum(1 for _ in open(target)) == len(jobs) and target.stat().st_mtime >= path.stat().st_mtime:
                results = [json.loads(line) for line in open(target)]
            else:
                results = pool.map(measure, jobs, chunksize=2)
                target.with_suffix(".tmp").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
                os.replace(target.with_suffix(".tmp"), target)
            sizes = sorted(r["tokens"] for r in results if r["tokens"])
            problems = {}
            for r in results:
                if not r["ok"]:
                    problems[r["problem"][:60]] = problems.get(r["problem"][:60], 0) + 1
            print(json.dumps(dict(rollout=path.stem, rows=len(results), ok=sum(r["ok"] for r in results), problems=problems,
                                  tokens=dict(p50=sizes[len(sizes) // 2], p90=sizes[int(0.9 * (len(sizes) - 1))], max=sizes[-1]) if sizes else {}),
                             ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
