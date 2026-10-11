#!/usr/bin/env python3
"""Qwen3.8-27B 新栈冒烟:transformers@main 的 qwen4_exp 类能不能正确算它。

背景:27B(Qwen3.8)在现行 5.14.1 栈下 step 2 deterministic NaN,根因是
qwen3_5 建模代码不认 output_gate_type=swish。上游 main 有 qwen4_exp
(qwen3_5 家族 VLM + gate 开关,gate 只收 {sigmoid, silu};swish≡silu)。

本 probe 做三件事,全部只读模型目录、不写盘:
  1. config 重映射(内存中):model_type → qwen4_exp,
     text_config.output_gate_type swish → silu;
  2. from_pretrained 加载权重,报 missing/unexpected keys(有 unexpected
     即键名不匹配,直接 FAIL);
  3. 对一小段 token 跑 CPU forward,验 logits 全有限。

用法:
    PYTHONPATH=<tf-main overlay>:$PYTHONPATH python3 probe_qwen38_forward.py <模型目录>

PASS 判据:无 unexpected keys 且 logits 有限。PASS 后才值得派生训练 yaml。
"""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    model_dir = sys.argv[1]
    import torch
    import transformers
    print("transformers", transformers.__version__, "torch", torch.__version__)

    raw = json.load(open(os.path.join(model_dir, "config.json")))
    print("原 model_type:", raw["model_type"],
          "gate:", raw["text_config"].get("output_gate_type"))

    from transformers import AutoConfig
    from transformers.models.qwen4_exp import Qwen4ExpConfig

    remapped = dict(raw)
    remapped["model_type"] = "qwen4_exp"
    remapped["text_config"] = dict(raw["text_config"])
    if remapped["text_config"].get("output_gate_type") == "swish":
        remapped["text_config"]["output_gate_type"] = "silu"
    cfg = Qwen4ExpConfig(**{k: v for k, v in remapped.items()
                            if k not in ("architectures", "transformers_version")})
    print("重映射后 gate:", cfg.text_config.output_gate_type)

    from transformers import AutoModelForImageTextToText
    import transformers.modeling_utils as mu
    model, loading_info = AutoModelForImageTextToText.from_pretrained(
        model_dir, config=cfg, dtype=torch.bfloat16,
        output_loading_info=True, low_cpu_mem_usage=True)
    missing = loading_info.get("missing_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    for k in (missing[:5] + unexpected[:5]):
        print("  ", k)
    if unexpected:
        print("PROBE_FAIL unexpected_keys —— 键名不匹配,qwen4_exp 不是这套权重的宿主")
        return 1

    model.eval()
    ids = torch.tensor([[raw["text_config"].get("bos_token_id") or 1,
                         5, 17, 91, 203, 1024]])
    with torch.no_grad():
        out = model(input_ids=ids)
    finite = torch.isfinite(out.logits).all().item()
    print("logits shape", tuple(out.logits.shape), "finite:", finite)
    print("PROBE_PASS" if finite else "PROBE_FAIL non-finite logits")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
