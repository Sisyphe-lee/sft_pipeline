#!/usr/bin/env python3
"""FLA chunk 系 backward buffer 的 zero-init 补丁(上游 PR #892 同型,wontfix 未合)。

背景:fla 0.5.1 的 gated delta rule chunk 路径里,backward(及部分前向)
buffer 用 `torch.empty_like` / `.new_empty` 分配,kernel 不保证写满每一行;
CUDA allocator 回收到带 NaN 的旧页时,垃圾值被 autograd 归约进权重梯度。
Qwen3.8-27B(48 v-heads / 16 qk-heads)在我们栈上 step1/2 backward 即中招,
step3 全 rank 前向 NaN(权重已NaN 写入)。上游 PR #892 实测同型修复把 NaN
参数从 ~400 归零;PR 被 wontfix 关闭,故自己打。

做法:对指定 FLA overlay 目录下三个文件做 empty→zeros 的机械替换:
    fla/ops/gated_delta_rule/wy_fast.py
    fla/ops/common/chunk_o.py
    fla/ops/common/chunk_delta_h.py
纯初始化语义,不改任何数学;代价是这几处分配多一次 memset。

用法:
    python3 patch_fla_zeroinit.py /path/to/qwendeps/<overlay>/
幂等:已打过(文件里已无 empty_like/new_empty)则报告 0 处并跳过。
原文件备份为 .orig_prezeroinit 后缀(已存在则不覆盖)。
"""
import os
import re
import shutil
import sys

FILES = [
    "fla/ops/gated_delta_rule/wy_fast.py",
    "fla/ops/common/chunk_o.py",
    "fla/ops/common/chunk_delta_h.py",
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = os.path.abspath(sys.argv[1])
    total = 0
    for rel in FILES:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            print(f"FAIL 缺文件 {path}")
            return 1
        src = open(path, "r", encoding="utf-8").read()
        patched = re.sub(r"\btorch\.empty_like\(", "torch.zeros_like(", src)
        patched = re.sub(r"\.new_empty\(", ".new_zeros(", patched)
        count = (src.count("torch.empty_like(") - patched.count("torch.empty_like(")) + \
                (src.count(".new_empty(") - patched.count(".new_empty("))
        if count:
            backup = path + ".orig_prezeroinit"
            if not os.path.exists(backup):
                shutil.copy2(path, backup)
            open(path, "w", encoding="utf-8").write(patched)
        print(f"{rel}: 替换 {count} 处")
        total += count
    print(f"PATCH_TOTAL={total}")
    print("PATCH_OK" if total >= 0 else "PATCH_FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
