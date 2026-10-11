#!/usr/bin/env python3
"""选数据：标签 + 策略 -> 训练清单。不调用模型，几秒完成，可以反复换策略。

    python3 data_pipeline/select_rows.py --policy configs/data_policies/pattern_all.yaml \\
        --rows DATA_DIR/rows --labels DATA_DIR/labels.jsonl [more.jsonl ...] [--tokens DATA_DIR/tokens] [--exclude fixes.jsonl] --output MANIFEST.json

清单是训练器读取的格式（data_pipeline/manifest.py）：按文件和字节位置指向训练行，不复制数据。
没选中的步不算 loss，但仍在后面步的历史里。
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import yaml

ARMS = ("labelled", "random_matched", "all")
POLICY = dict(arm="labelled", patterns=[], seed=0, require_reasoning=True, max_tokens=196608)


def index_rows(path, tokens_path, require_reasoning, max_tokens):
    """([(turn, offset, length, tokens)] of the rows that may carry loss, {why not: count} for the others, task identity)."""
    measured = [json.loads(line) for line in open(tokens_path)] if tokens_path else None
    out, unusable, offset, task = [], {}, 0, None
    with open(path, "rb") as stream:
        for n, line in enumerate(stream):
            meta = json.loads(line)["metadata"]
            task = task or meta["task"]
            usage = meta.get("usage") or {}
            tokens = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0) or max(1, len(line) // 4)
            why = None
            if meta.get("kind", "turn") != "turn":
                why = "not an agent turn"
            elif require_reasoning and "reasoning" not in meta.get("holds", ()):
                why = "reply without reasoning"
            elif measured is not None:
                if n >= len(measured) or measured[n]["turn"] != meta["turn"]:
                    raise ValueError(f"{tokens_path} does not match {path}; run measure_rows.py again")
                tokens = measured[n]["tokens"] or tokens
                if not measured[n]["ok"]:
                    why = "trainer cannot use it: " + measured[n]["problem"][:50]
                elif max_tokens and tokens > max_tokens:
                    why = "longer than max_tokens"
            if why:
                unusable[why] = unusable.get(why, 0) + 1
            else:
                out.append((meta["turn"], offset, len(line), tokens))
            offset += len(line)
    return out, unusable, task


def select(rows_dir, label_paths, output, arm="labelled", patterns=(), seed=0, exclude_paths=(), tokens_dir=None,
           require_reasoning=True, max_tokens=196608):
    """arm `labelled`: every usable row a label covers, once. `random_matched`: from each rollout as many rows as
    `labelled` takes, drawn at random among its usable rows. `all`: every usable row. A row is usable when it is an
    agent turn, its reply has reasoning (`require_reasoning`) and, with `tokens_dir` (measure_rows.py), the trainer
    accepts it and it is no longer than `max_tokens`."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    rows_dir, output = Path(rows_dir), Path(output)
    wanted, per_pattern, digests = {}, {}, {}
    for path in map(Path, label_paths):
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            label = json.loads(line)
            if patterns and label["pattern"] not in patterns:
                continue
            per_pattern[label["pattern"]] = per_pattern.get(label["pattern"], 0) + 1
            wanted.setdefault(label["trajectory_id"], set()).update(label["steps"])
    dropped = 0
    for path in map(Path, exclude_paths):  # corrections: {"trajectory_id": ..., "steps": [...]} lines take steps out again
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            fix = json.loads(line)
            held = wanted.get(fix["trajectory_id"], set())
            dropped += len(held & set(fix["steps"]))
            held -= set(fix["steps"])
    names = sorted(p.stem for p in rows_dir.glob("*.jsonl")) if arm == "all" else sorted(wanted)
    rng = random.Random(seed)
    sources, chosen, stats, unusable_all = [], [], {}, {}
    for trajectory in names:
        tokens_path = Path(tokens_dir) / f"{trajectory}.jsonl" if tokens_dir else None
        if tokens_path and not tokens_path.exists():
            raise FileNotFoundError(f"{tokens_path}: run measure_rows.py on this rollout first")
        rows, unusable, task = index_rows(rows_dir / f"{trajectory}.jsonl", tokens_path, require_reasoning, max_tokens)
        labelled = [r for r in rows if r[0] in wanted.get(trajectory, ())]
        picked = labelled if arm == "labelled" else rows if arm == "all" else sorted(rng.sample(rows, len(labelled)))
        stats[trajectory] = dict(usable=len(rows), unusable=unusable, labelled=len(wanted.get(trajectory, ())), labelled_usable=len(labelled), selected=len(picked))
        for why, count in unusable.items():
            unusable_all[why] = unusable_all.get(why, 0) + count
        if not picked:
            continue
        sources.append(f"{trajectory}.jsonl")
        keys = dict(task_id=str(task["id"]), trajectory_id=trajectory, split_group=json.dumps([str(task["id"]), task.get("seed", 0)], separators=(",", ":")))
        chosen += [dict(source=len(sources) - 1, source_line=r[0], offset=r[1], length=r[2], estimated_tokens=r[3], **keys) for r in picked]
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(version=2, base_root=os.path.relpath(rows_dir.resolve(), output.parent.resolve()), sources=sources, train=chosen, validation=[],
                    selection=dict(arm=arm, patterns=list(patterns) or "all", seed=seed, require_reasoning=require_reasoning, max_tokens=max_tokens,
                                   lengths="measured" if tokens_dir else "not measured", labels=digests, labelled_spans=per_pattern, excluded_steps=dropped,
                                   rows=len(chosen), labelled_steps=sum(s["labelled"] for s in stats.values()),
                                   labelled_usable=sum(s["labelled_usable"] for s in stats.values()), unusable=unusable_all, rollouts=stats))
    output.with_suffix(".tmp").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    os.replace(output.with_suffix(".tmp"), output)
    return manifest["selection"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, type=Path, help="configs/data_policies/*.yaml")
    parser.add_argument("--rows", required=True, type=Path, help="directory of training rows written by annotate.py")
    parser.add_argument("--labels", nargs="*", default=[], type=Path, help="label files written by annotate.py")
    parser.add_argument("--tokens", type=Path, help="directory written by measure_rows.py; without it lengths are not checked")
    parser.add_argument("--exclude", nargs="*", default=[], type=Path,
                        help="correction files: lines {\"trajectory_id\": ..., \"steps\": [...]} whose steps are not selected")
    parser.add_argument("--output", required=True, type=Path, help="the training manifest to write (.json)")
    args = parser.parse_args()
    policy = dict(POLICY, **(yaml.safe_load(args.policy.read_text()) or {}))
    if policy.keys() - POLICY.keys():
        parser.error(f"unknown policy keys: {sorted(policy.keys() - POLICY.keys())}")
    summary = select(args.rows, args.labels, args.output, policy["arm"], policy["patterns"], policy["seed"], args.exclude, args.tokens,
                     policy["require_reasoning"], policy["max_tokens"])
    print(f"manifest {args.output}: arm {summary['arm']}, {summary['rows']} rows from {len(summary['rollouts'])} rollouts; "
          f"labels cover {summary['labelled_steps']} steps, {summary['labelled_usable']} usable; unusable rows {summary['unusable'] or 'none'}; "
          f"lengths {summary['lengths']}; labelled spans {summary['labelled_spans']}")


if __name__ == "__main__":
    main()
