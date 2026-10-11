#!/usr/bin/env python3
"""从训练的 master.log 里把逐步 metrics 提成一份小 JSON。

HF Trainer 每个 logging step 打一行 `{'loss': '0.9498', 'grad_norm': ...}`,混在
进度条的回车与 ANSI 里。这里按行扫,把出现的字段都取出来 —— 不同 TRL 版本与不同
loss 归一路径打的字段不一样,有哪个提哪个:

- 全都有的:loss、grad_norm、learning_rate、epoch
- 有的 run 才有:entropy、num_tokens、mean_token_accuracy

2026-09-11 那次 27B 训练(train_multi_lcyfull_base27b_mn4_noovl_gbl)1938 条记录里
一条 mean_token_accuracy 都没有,而 2026-09-10 的 c910 那次 1158 条全带。所以字段
缺失是那次 run 本身没记,不是这个脚本没提到。

一份 411 MB 的 master.log 提完大约 100 KB,可以直接回传。

用法:
  python3 training/tools/extract_train_metrics.py <master.log> [--out metrics.json]
  python3 training/tools/extract_train_metrics.py <master.log> --fields loss,mean_token_accuracy
"""

from __future__ import annotations

import argparse
import json
import re
import sys

FIELDS = ("loss", "grad_norm", "learning_rate", "entropy",
          "num_tokens", "mean_token_accuracy", "epoch")
# 行里是 Python dict 的 repr,值带引号。逐字段单独抓,顺序和缺失都不影响。
PAIR = re.compile(r"'(%s)':\s*'([\d.eE+-]+)'" % "|".join(FIELDS))
ROW = re.compile(r"\{'loss':.*?\}")


def extract(path, want=None):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "'loss'" not in line:
                continue
            for chunk in ROW.findall(line):
                got = {k: float(v) for k, v in PAIR.findall(chunk)}
                if "loss" not in got:
                    continue
                rows.append({k: v for k, v in got.items() if not want or k in want})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", help="写到这个文件;不给就打到 stdout")
    ap.add_argument("--fields", help="逗号分隔,只要这几个字段")
    a = ap.parse_args()

    want = set(a.fields.split(",")) if a.fields else None
    rows = extract(a.log, want)
    if not rows:
        print("ABORT: 这份日志里一条 loss 记录都没有", file=sys.stderr)
        return 2

    present = sorted({k for r in rows for k in r})
    text = json.dumps(rows, separators=(",", ":"))
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
        print("%s:%d 条,字段 %s,%d 字节" % (a.out, len(rows), present, len(text)))
    else:
        print("N=%d FIELDS=%s" % (len(rows), ",".join(present)), file=sys.stderr)
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
