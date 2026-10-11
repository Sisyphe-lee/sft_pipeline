#!/usr/bin/env python3
"""Turn labels into training manifests: the labelled rows, and a control of as many random rows from the same rollouts.

    select_rows.py --rows DIR --labels labels.jsonl --out PREFIX [--patterns C1 D3 ...] [--seed 0]

Writes PREFIX-pattern.json and PREFIX-control.json in the manifest format the trainer reads (data_pipeline/manifest.py):
rows are named by source file and byte offset, nothing is copied. A row is in the pattern manifest when a label covers
its step; a row that carries several labels is listed once. The control manifest takes, from each rollout, the same
number of rows at random among all its ordinary turns. Rows that are not selected still appear as history inside
later rows; they just carry no loss.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def index_rows(path):
    """[(turn, offset, length, estimated tokens, kind)] of one export file."""
    out, offset = [], 0
    with open(path, "rb") as stream:
        for line in stream:
            meta = json.loads(line)["metadata"]
            usage = meta.get("usage") or {}
            tokens = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0) or max(1, len(line) // 4)
            out.append((meta["turn"], offset, len(line), tokens, meta.get("kind", "turn")))
            offset += len(line)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", required=True, type=Path, help="directory of export files, <trajectory id>.jsonl")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, help="path prefix of the two manifests")
    parser.add_argument("--patterns", nargs="+", help="use only these patterns (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the control sample")
    args = parser.parse_args()

    wanted, per_pattern = {}, {}
    for line in args.labels.read_text().splitlines():
        label = json.loads(line)
        if args.patterns and label["pattern"] not in args.patterns:
            continue
        per_pattern[label["pattern"]] = per_pattern.get(label["pattern"], 0) + 1
        for step in label["steps"]:
            wanted.setdefault(label["trajectory_id"], {}).setdefault(step, set()).add(label["pattern"])
    rng = random.Random(args.seed)
    sources, chosen, control, stats = [], [], [], {}
    for trajectory in sorted(wanted):
        path = args.rows / f"{trajectory}.jsonl"
        rows = [r for r in index_rows(path) if r[4] == "turn"]
        task = json.loads(path.open().readline())["metadata"]["task"]
        keys = dict(task_id=str(task["id"]), trajectory_id=trajectory, split_group=json.dumps([str(task["id"]), task.get("seed", 0)], separators=(",", ":")))
        sources.append(path.name)

        def entry(row):
            return dict(source=len(sources) - 1, source_line=row[0], offset=row[1], length=row[2], estimated_tokens=row[3], **keys)

        picked = [r for r in rows if r[0] in wanted[trajectory]]
        chosen += [entry(r) for r in picked]
        control += [entry(r) for r in sorted(rng.sample(rows, len(picked)))]
        stats[trajectory] = dict(turns=len(rows), labelled=len(picked))
    digest = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    for name, rows in (("pattern", chosen), ("control", control)):
        target = Path(f"{args.out}-{name}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest = dict(version=2, base_root=os.path.relpath(args.rows.resolve(), target.parent.resolve()), sources=sources, train=rows, validation=[],
                        selection=dict(arm=name, labels=args.labels.name, labels_sha256=digest, patterns=args.patterns or "all", control_seed=args.seed,
                                       labelled_spans=per_pattern, rows=len(rows), rollouts=stats))
        target.with_suffix(".tmp").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        os.replace(target.with_suffix(".tmp"), target)
    print(f"{len(chosen)} rows in each manifest from {len(sources)} rollouts; labelled spans per pattern {dict(sorted(per_pattern.items()))}")


if __name__ == "__main__":
    main()
