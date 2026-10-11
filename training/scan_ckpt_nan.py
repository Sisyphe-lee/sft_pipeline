#!/usr/bin/env python3
"""扫 checkpoint 里的非有限参数(27B NaN 定位的离线扫描器)。

对 safetensors 分片逐个加载(CPU),统计每个 tensor 的 NaN/Inf 元素数,
按模块归类汇总 —— 哪些层的参数变成 NaN 一目了然。

用法:
    python3 scan_ckpt_nan.py <checkpoint 目录>
输出:逐 tensor 的非有限计数(只列受污染的)+ 按层型汇总 + SCAN_DONE。
"""
import collections
import glob
import json
import os
import sys


def bucket(name: str) -> str:
    for key in ("linear_attn", "self_attn", "mlp", "embed", "lm_head",
                "visual", "norm"):
        if key in name:
            return key
    return "other"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    import torch
    from safetensors.torch import load_file
    ckpt = os.path.abspath(sys.argv[1])
    shards = sorted(glob.glob(os.path.join(ckpt, "model*.safetensors")))
    assert shards, f"没有 safetensors 分片:{ckpt}"
    per_bucket = collections.Counter()
    bucket_total = collections.Counter()
    bad_tensors = 0
    total_tensors = 0
    for shard in shards:
        tensors = load_file(shard)
        for name, t in tensors.items():
            total_tensors += 1
            b = bucket(name)
            bucket_total[b] += 1
            if t.is_floating_point():
                nonfinite = int((~torch.isfinite(t)).sum().item())
                if nonfinite:
                    bad_tensors += 1
                    per_bucket[b] += 1
                    print(f"BAD {name} nonfinite={nonfinite}/{t.numel()}")
        del tensors
    print(f"\nSUMMARY tensors={total_tensors} bad={bad_tensors}")
    for b in sorted(bucket_total):
        print(f"  {b}: {per_bucket.get(b, 0)}/{bucket_total[b]} 受污染")
    print("SCAN_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
