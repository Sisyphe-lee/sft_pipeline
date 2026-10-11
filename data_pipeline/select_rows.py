#!/usr/bin/env python3
"""选数据：标签 + 策略 -> 训练清单。不调用模型，几秒完成，可以反复换策略。

    python3 data_pipeline/select_rows.py --policy configs/data_policies/pattern_all.yaml \\
        --rows DATA_DIR/rows --labels DATA_DIR/labels.jsonl [more.jsonl ...] [--exclude fixes.jsonl] --output MANIFEST.json

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


def index_rows(path):
    """[(turn, offset, length, estimated tokens)] of the ordinary turns of one export file, and its task identity."""
    out, offset, task = [], 0, None
    with open(path, "rb") as stream:
        for line in stream:
            meta = json.loads(line)["metadata"]
            task = task or meta["task"]
            usage = meta.get("usage") or {}
            tokens = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0) or max(1, len(line) // 4)
            if meta.get("kind", "turn") == "turn":
                out.append((meta["turn"], offset, len(line), tokens))
            offset += len(line)
    return out, task


def select(rows_dir, label_paths, output, arm="labelled", patterns=(), seed=0, exclude_paths=()):
    """arm `labelled`: every row a label covers, once. `random_matched`: from each rollout as many rows as `labelled`
    would take, drawn at random among its ordinary turns. `all`: every ordinary turn."""
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
    sources, chosen, stats = [], [], {}
    for trajectory in names:
        rows, task = index_rows(rows_dir / f"{trajectory}.jsonl")
        labelled = [r for r in rows if r[0] in wanted.get(trajectory, ())]
        picked = labelled if arm == "labelled" else rows if arm == "all" else sorted(rng.sample(rows, len(labelled)))
        stats[trajectory] = dict(turns=len(rows), labelled=len(labelled), selected=len(picked))
        if not picked:
            continue
        sources.append(f"{trajectory}.jsonl")
        keys = dict(task_id=str(task["id"]), trajectory_id=trajectory, split_group=json.dumps([str(task["id"]), task.get("seed", 0)], separators=(",", ":")))
        chosen += [dict(source=len(sources) - 1, source_line=r[0], offset=r[1], length=r[2], estimated_tokens=r[3], **keys) for r in picked]
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(version=2, base_root=os.path.relpath(rows_dir.resolve(), output.parent.resolve()), sources=sources, train=chosen, validation=[],
                    selection=dict(arm=arm, patterns=list(patterns) or "all", seed=seed, labels=digests, labelled_spans=per_pattern, excluded_steps=dropped,
                                   rows=len(chosen), rollouts=stats))
    output.with_suffix(".tmp").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    os.replace(output.with_suffix(".tmp"), output)
    return manifest["selection"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, type=Path, help="configs/data_policies/*.yaml")
    parser.add_argument("--rows", required=True, type=Path, help="directory of training rows written by annotate.py")
    parser.add_argument("--labels", nargs="*", default=[], type=Path, help="label files written by annotate.py")
    parser.add_argument("--exclude", nargs="*", default=[], type=Path,
                        help="correction files: lines {\"trajectory_id\": ..., \"steps\": [...]} whose steps are not selected")
    parser.add_argument("--output", required=True, type=Path, help="the training manifest to write (.json)")
    args = parser.parse_args()
    policy = dict(dict(arm="labelled", patterns=[], seed=0), **(yaml.safe_load(args.policy.read_text()) or {}))
    if policy.keys() - {"arm", "patterns", "seed"}:
        parser.error(f"unknown policy keys: {sorted(policy.keys() - {'arm', 'patterns', 'seed'})}")
    summary = select(args.rows, args.labels, args.output, policy["arm"], policy["patterns"], policy["seed"], args.exclude)
    print(f"manifest {args.output}: arm {summary['arm']}, {summary['rows']} rows from {len(summary['rollouts'])} rollouts, labelled spans {summary['labelled_spans']}")


if __name__ == "__main__":
    main()
