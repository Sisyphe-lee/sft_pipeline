"""Write a training manifest from labels: which rows of the export carry loss.

Used by clean.py. A manifest is in the format the trainer reads (data_pipeline/manifest.py): rows are named by
source file and byte offset, nothing is copied. Rows that are not selected still appear as history inside later rows.
"""
import hashlib
import json
import os
import random
from pathlib import Path

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


def select(rows_dir, labels_path, output, arm="labelled", patterns=(), seed=0, trajectories=None):
    """arm `labelled`: every row a label covers, once. `random_matched`: from each rollout as many rows as `labelled`
    would take, drawn at random among its ordinary turns. `all`: every ordinary turn."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    rows_dir, labels_path, output = Path(rows_dir), Path(labels_path), Path(output)
    wanted, per_pattern = {}, {}
    for line in labels_path.read_text().splitlines():
        label = json.loads(line)
        if patterns and label["pattern"] not in patterns:
            continue
        per_pattern[label["pattern"]] = per_pattern.get(label["pattern"], 0) + 1
        wanted.setdefault(label["trajectory_id"], set()).update(label["steps"])
    names = sorted(trajectories if trajectories is not None else wanted)
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
                    selection=dict(arm=arm, patterns=list(patterns) or "all", seed=seed, labels=labels_path.name,
                                   labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(), labelled_spans=per_pattern,
                                   rows=len(chosen), rollouts=stats))
    output.with_suffix(".tmp").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    os.replace(output.with_suffix(".tmp"), output)
    return manifest["selection"]
