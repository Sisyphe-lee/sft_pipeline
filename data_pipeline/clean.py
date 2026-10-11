#!/usr/bin/env python3
"""数据清洗的唯一入口：抓包 -> 训练行 -> 标注 -> 训练清单。

    python3 data_pipeline/clean.py --policy configs/data_policies/pattern_all.yaml \\
        --capture CAPTURE_DIR [CAPTURE_DIR ...] --runs RUNS_DIR --tasks TASKS_DIR --data DATA_DIR --output MANIFEST.json

做什么由 policy 文件决定（configs/data_policies/）。已完成的步骤会跳过，中断后重跑同一条命令即可；
换一个只改了选择方式的 policy 时，导出和标注都直接复用。详见 README.md。
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_capture  # noqa: E402
import label  # noqa: E402
import rollout  # noqa: E402
import select_rows  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULTS = dict(
    arm="labelled", patterns=[], seed=0,
    labelling=dict(sheet="patterns/label-sheet.md", readings=["timeline", "ideas", "surprises"], per_url=2, window_chars=150000,
                   read_max_tokens=40000, judge_max_tokens=30000, render_cap=3000),
)


def load_policy(path):
    given = yaml.safe_load(Path(path).read_text()) or {}
    unknown = (given.keys() - DEFAULTS.keys()) | ((given.get("labelling") or {}).keys() - DEFAULTS["labelling"].keys())
    if unknown:
        raise ValueError(f"unknown policy keys: {sorted(unknown)}")
    policy = dict(DEFAULTS, **{k: v for k, v in given.items() if k != "labelling"})
    policy["labelling"] = dict(DEFAULTS["labelling"], **(given.get("labelling") or {}))
    return policy


def kimi_urls():
    if os.environ.get("KIMI_URLS"):
        return os.environ["KIMI_URLS"].split()
    out = subprocess.run(["bash", "-c", "source /personal/kimi.env && echo $KIMI_BASE_URL_1 $KIMI_BASE_URL_2 $KIMI_BASE_URL_3 $KIMI_BASE_URL_4"],
                         capture_output=True, text=True).stdout.split()
    if not out:
        sys.exit("no labelling endpoint: set KIMI_URLS or provide /personal/kimi.env")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, type=Path, help="configs/data_policies/*.yaml")
    parser.add_argument("--capture", nargs="+", required=True, help="bridge capture directories (resp_<id>/...)")
    parser.add_argument("--runs", required=True, type=Path, help="harness run directories: RUNS/<trajectory id>/<task>/")
    parser.add_argument("--tasks", type=Path, help="EdgeBench task specifications (<task>.json), for the score direction")
    parser.add_argument("--data", required=True, type=Path, help="working directory: rows/, work/, labels-*.jsonl are kept here")
    parser.add_argument("--output", required=True, type=Path, help="the training manifest to write (.json)")
    parser.add_argument("--only", help="use only rollouts whose capture id contains this text")
    args = parser.parse_args()
    policy = load_policy(args.policy)
    lab = policy["labelling"]
    rows_dir, work = args.data / "rows", args.data / "work"
    rows_dir.mkdir(parents=True, exist_ok=True)

    # 1. export: one row per model reply
    names = []
    for ident, exchanges in sorted(export_capture.scan(args.capture, args.only).items()):
        task, run = export_capture.identity(ident)
        names.append((task, run))
        if not (rows_dir / f"{run}.jsonl").exists():
            print("export", json.dumps(export_capture.export(ident, exchanges, rows_dir), ensure_ascii=False), flush=True)
    # 2. render: numbered steps and the places a rule finds
    for task, run in names:
        if not (work / run / "points.json").exists():
            run_dir = args.runs / run / task
            r = rollout.Rollout(rows_dir / f"{run}.jsonl", run_dir if run_dir.is_dir() else None, args.tasks)
            r.render(work / run, lab["render_cap"])
            print(f"render {run}: {len(r.steps)} steps, {len(r.subs)} judged rounds" + ("" if run_dir.is_dir() else " (no run directory found)"), flush=True)
    dirs = [work / run for _, run in names]
    if policy["arm"] != "all":
        # 3. readings and 4. places: the labelling model reports where to look
        urls = kimi_urls()
        todo, lock = list(dirs), threading.Lock()

        def reader(url):
            while True:
                with lock:
                    if not todo:
                        return
                    d = todo.pop(0)
                for k, name in enumerate(lab["readings"]):
                    label.read(argparse.Namespace(dir=str(d), protocol=str(REPO / f"patterns/{name}-protocol.md"), url=url, name=name,
                                                  max_tokens=lab["read_max_tokens"], temperature=0.0, timeout=3600, tries=3, extra="{}", force=False,
                                                  segment_chars=1200000, context=f"{lab['readings'][0]}.json" if k else None))
                if not (d / "places.json").exists():
                    label.places(argparse.Namespace(dir=str(d), timeline=[f"{n}.json" for n in lab["readings"][1:] + lab["readings"][:1]], batch=1))

        threads = [threading.Thread(target=reader, args=(u,)) for u in urls for _ in range(lab["per_url"])]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 5. label: the model answers the sheet's questions for every place; label.py decides
        sheet = REPO / lab["sheet"]
        version = hashlib.sha256(sheet.read_bytes()).hexdigest()[:8]
        label.judge(argparse.Namespace(dirs=[str(d) for d in dirs], sheet=[str(sheet)], url=urls, per_url=lab["per_url"], name=f"judged-{version}",
                                       packets="packets", timeline=f"{lab['readings'][0]}.json", window_chars=lab["window_chars"],
                                       max_tokens=lab["judge_max_tokens"], temperature=0.0, timeout=3600, tries=3, extra="{}"))
        labels = args.data / f"labels-{version}.jsonl"
        label.labels(argparse.Namespace(dirs=[str(d) for d in dirs], name=f"judged-{version}", out=str(labels)))
    else:
        labels = args.data / "labels-none.jsonl"
        labels.touch()
    # 6. select: the manifest
    summary = select_rows.select(rows_dir, labels, args.output, policy["arm"], policy["patterns"], policy["seed"], [run for _, run in names])
    print(f"manifest {args.output}: arm {summary['arm']}, {summary['rows']} rows from {len(names)} rollouts, labelled spans {summary['labelled_spans']}")


if __name__ == "__main__":
    main()
