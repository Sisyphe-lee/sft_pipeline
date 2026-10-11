#!/usr/bin/env python3
"""单行 forward 诊断:指定训练行在指定模型上的 loss 是否有限。

用途:27B(Qwen3.8)正式训练 step3 八 rank 同步 NaN 的判别实验 B ——
在 CPU 纯 torch 路径(无 FLA fused kernel、无 GPU bf16 kernel)上 forward
已知 NaN 行。结果解读:
  - CPU 也 NaN → 权重/结构层面的问题(与 kernel 无关);
  - CPU 有限 → 嫌疑收窄到 GPU kernel 路径(FLA/flash-attn/bf16 内核)。
配合 --dtype fp32 可再分一层:bf16 NaN 而 fp32 有限 = 数值范围问题。

输入组装逐行复刻 train_sft.py 的 collate_vlm(chat template 全串 + prompt
前缀、completion-only labels),保证测的就是训练看到的那份输入。

用法:
    PYTHONPATH=<train overlay> python3 diag_row_forward.py \
        --model <dir> --dataset <jsonl> --row 1924 [--dtype bf16|fp32]
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cuda = 单卡 GPU 前向,走真实 fused kernel 栈")
    ap.add_argument("--hooks", action="store_true",
                    help="逐模块挂 forward hook,报告第一个产出非有限值的模块")
    ap.add_argument("--backward", action="store_true",
                    help="train 模式跑 forward+backward,检查全部参数梯度有限性;"
                         "配合 full_backward_hook 点名第一个产出非有限 grad 的模块。"
                         "背景:lr=0 挡不住 NaN 梯度(0×NaN=NaN 把 NaN 写进权重),"
                         "所以'哪步 backward 先产生 NaN 梯度'才是真判据")
    args = ap.parse_args()

    # CPU 模式:把 CUDA-only 的内核包全部藏掉,逼 modeling 走 torch 回退
    # (causal_conv1d → F.conv1d、gated delta rule → torch 实现、
    # RMSNormGated → 纯 torch 版)。sys.modules 置 None 会让 import 直接
    # ImportError,正中 modeling 里 try/except 的 except 分支。
    # cuda 模式不藏 —— 就是要测真实 fused kernel 栈。
    if args.device == "cpu":
        for _mod in ("causal_conv1d", "fla", "flash_attn"):
            sys.modules[_mod] = None
    import torch
    from PIL import Image
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from physical_agent.artifact_refs import resolve_artifact_ref
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import transformers
    print("transformers", transformers.__version__, flush=True)

    row = None
    with open(args.dataset, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i == args.row:
                row = json.loads(line)
                break
    assert row is not None, f"row {args.row} 不存在"
    dataset_root = os.path.dirname(os.path.abspath(args.dataset))

    processor = AutoProcessor.from_pretrained(args.model)
    images = [Image.open(resolve_artifact_ref(p, root=dataset_root)).convert("RGB")
              for p in row["images"]]
    full_text = processor.apply_chat_template(
        row["prompt"] + row["completion"], tokenize=False,
        add_generation_prompt=False, tools=row.get("tools"))
    prompt_text = processor.apply_chat_template(
        row["prompt"], tokenize=False,
        add_generation_prompt=True, tools=row.get("tools"))
    inputs = processor(text=[full_text], images=images, return_tensors="pt", padding=True)
    prompt_len = processor(text=[prompt_text], images=images,
                           return_tensors="pt", padding=True)["input_ids"].shape[1]
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    print(f"row={args.row} seq={inputs['input_ids'].shape[1]} "
          f"prompt_len={prompt_len} images={len(images)}", flush=True)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    attn_impl = "sdpa" if args.device == "cpu" else "flash_attention_2"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation=attn_impl)
    model.eval()
    if args.device == "cuda":
        model.cuda()
        inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}
        labels = labels.cuda()

    first_bad = []
    if args.hooks:
        def make_hook(name):
            def hook(_m, _inp, out_):
                if first_bad:
                    return
                ts = out_ if isinstance(out_, torch.Tensor) else (
                    out_[0] if isinstance(out_, tuple) and isinstance(out_[0], torch.Tensor) else None)
                if ts is not None and ts.is_floating_point() and not torch.isfinite(ts).all():
                    first_bad.append(name)
                    print(f"FIRST_NONFINITE_MODULE {name} shape={tuple(ts.shape)}", flush=True)
            return hook
        for name, mod in model.named_modules():
            if name:
                mod.register_forward_hook(make_hook(name))

    if args.backward:
        # backward 判据路径:train 模式建图,反向后查每个参数梯度。
        # grad hook 在反向经过每个模块时看 grad_output,第一个非有限即点名
        # (named 反向顺序从 lm_head 往下,点到的名字是"NaN 来源下游第一站",
        # 配合参数 grad 扫描能夹出肇事层)。
        model.train()
        model.gradient_checkpointing_enable()
        first_bad_grad = []
        def make_bhook(name):
            def bhook(_m, gin, gout):
                if first_bad_grad:
                    return
                for g in (list(gout) if isinstance(gout, tuple) else [gout]):
                    if isinstance(g, torch.Tensor) and g.is_floating_point() \
                            and not torch.isfinite(g).all():
                        first_bad_grad.append(name)
                        print(f"FIRST_NONFINITE_GRAD_AT {name}", flush=True)
                        return
            return bhook
        for name, mod in model.named_modules():
            if name:
                mod.register_full_backward_hook(make_bhook(name))
        out = model(**inputs, labels=labels)
        out.loss.backward()
        bad_params = [n for n, p in model.named_parameters()
                      if p.grad is not None and not torch.isfinite(p.grad).all()]
        print(f"nonfinite_grad_params={len(bad_params)}", flush=True)
        for n in bad_params[:8]:
            print("  BAD_GRAD", n, flush=True)
        loss_f = float(out.loss)
        finite = torch.isfinite(out.loss).item() and not bad_params
        print(json.dumps({"row": args.row, "mode": "backward", "loss": loss_f,
                          "finite": bool(finite)}), flush=True)
        print("DIAG_BWD_FINITE" if finite else "DIAG_BWD_NONFINITE", flush=True)
        return 0 if finite else 1

    with torch.no_grad():
        out = model(**inputs, labels=labels)
    loss = float(out.loss)
    logit_max = float(out.logits.abs().max())
    finite = torch.isfinite(out.loss).item() and torch.isfinite(out.logits).all().item()
    print(json.dumps({"row": args.row, "dtype": args.dtype, "loss": loss,
                      "logits_absmax": logit_max, "finite": bool(finite)}), flush=True)
    print("DIAG_FINITE" if finite else "DIAG_NONFINITE", flush=True)
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
