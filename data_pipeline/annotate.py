#!/usr/bin/env python3
"""标注：抓包 -> 训练行 -> Kimi 标注 -> 标签文件。离线做一次，之后选数据时反复使用。

    python3 data_pipeline/annotate.py --config configs/labelling/kimi_v1.yaml \\
        --capture CAPTURE_DIR [CAPTURE_DIR ...] --runs RUNS_DIR --tasks TASKS_DIR --data DATA_DIR

写出 DATA_DIR/rows/（训练行）和 DATA_DIR/labels.jsonl（标签）。只问还没问过的：中断后重跑即可，
新增 rollout 只标新增的，改了某个 patterns/<名字>.yaml 只重标这一个 pattern。选数据见 select_rows.py。
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_capture  # noqa: E402
import label  # noqa: E402
import rollout  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULTS = dict(patterns=[], readings=["timeline", "ideas", "surprises"], per_url=2, window_chars=150000,
                read_max_tokens=40000, judge_max_tokens=30000, render_cap=3000)


def load_config(path):
    given = yaml.safe_load(Path(path).read_text()) or {}
    if given.keys() - DEFAULTS.keys():
        raise ValueError(f"unknown config keys: {sorted(given.keys() - DEFAULTS.keys())}")
    return dict(DEFAULTS, **given)


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
    parser.add_argument("--config", required=True, type=Path, help="configs/labelling/*.yaml")
    parser.add_argument("--capture", nargs="+", required=True, help="bridge capture directories (resp_<id>/...)")
    parser.add_argument("--runs", required=True, type=Path, help="harness run directories: RUNS/<trajectory id>/<task>/")
    parser.add_argument("--tasks", type=Path, help="EdgeBench task specifications (<task>.json), for the score direction")
    parser.add_argument("--data", required=True, type=Path, help="data directory: rows/, work/ and labels-*.jsonl are written here")
    parser.add_argument("--only", help="use only rollouts whose capture id contains this text")
    args = parser.parse_args()
    lab = load_config(args.config)
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
    # 3. readings and 4. places: the labelling model reports where to look
    urls = kimi_urls()
    todo, lock = list(dirs), threading.Lock()

    failed = []

    def reader(url):
        while True:
            with lock:
                if not todo or failed:
                    return
                d = todo.pop(0)
            try:
                read_one(url, d)
            except BaseException as error:  # noqa: BLE001 - a thread must not die silently; report and stop the run
                failed.append(f"{d.name}: {type(error).__name__} {error}")

    def read_one(url, d):
        for k, name in enumerate(lab["readings"]):
            label.read(argparse.Namespace(dir=str(d), protocol=str(REPO / f"patterns/{name}-protocol.md"), url=url, name=name,
                                          max_tokens=lab["read_max_tokens"], temperature=0.0, timeout=3600, tries=3, extra="{}", force=False,
                                          segment_chars=1200000, context=f"{lab['readings'][0]}.json" if k else None))
        if not (d / "places.json").exists():
            label.places(argparse.Namespace(dir=str(d), timeline=[f"{n}.json" for n in lab["readings"][1:] + lab["readings"][:1]]))


    threads = [threading.Thread(target=reader, args=(u,)) for u in urls for _ in range(lab["per_url"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        sys.exit("reading failed: " + "; ".join(failed))
    # 5. label: one question per place and pattern; only what the current pattern files have not been asked yet
    label.judge(argparse.Namespace(dirs=[str(d) for d in dirs], url=urls, per_url=lab["per_url"], patterns=lab["patterns"],
                                   timeline=f"{lab['readings'][0]}.json", window_chars=lab["window_chars"],
                                   max_tokens=lab["judge_max_tokens"], temperature=0.0, timeout=3600, tries=3, extra="{}"))
    # 6. the labels of the current pattern versions; the file before this run is kept beside it
    labels = args.data / "labels.jsonl"
    if labels.exists():
        labels.replace(args.data / f"labels.before-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    label.labels(argparse.Namespace(dirs=[str(d) for d in dirs], out=str(labels)))
    print(f"labels: {labels}")


if __name__ == "__main__":
    main()
