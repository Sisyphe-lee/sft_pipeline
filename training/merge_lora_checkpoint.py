#!/usr/bin/env python3
"""把 LoRA adapter 合进底模,产出一个"长得跟全量 finetune 一样"的 checkpoint。

    python3 merge_lora_checkpoint.py --base <底模目录> \
        --adapter <checkpoint-1730> --output <merged 目录>

为什么要这一步:训练开 ``--lora`` 之后,trainer 存的是
``adapter_model.safetensors`` + ``adapter_config.json``,而 eval 那条通路
(``run_qwen_file_turn_eval.py`` 以及 pinned checkout 里的同名文件)加载的是
一个完整模型目录。与其去改 eval —— 那会动到已经钉死的 execution lineage,
把"LoRA vs 全量"这个自变量和"eval 代码变了"混在一起 —— 不如在训练之后显式
合并一次。合并出来的目录与全量 finetune 的 checkpoint 同构,eval 一行都不用改。

合并是显式的一步而不是训练里的自动行为,因为它有损:merge 之后就分不出
哪些权重来自 adapter 了。想保留可插拔的 adapter,就留着原目录,另存 merged。

注意 dtype:底模是 bfloat16 加载的,合并在 bf16 上做,与全量 finetune 存出来
的 checkpoint 精度一致。这里不做 fp32 合并 —— 那会让两条线的 checkpoint
精度不同,又多一个变量。
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path, help="底模目录")
    parser.add_argument("--adapter", required=True, type=Path, help="含 adapter 的 checkpoint 目录")
    parser.add_argument("--output", required=True, type=Path, help="合并后的目录(必须不存在)")
    args = parser.parse_args()

    adapter_cfg = args.adapter / "adapter_config.json"
    if not adapter_cfg.is_file():
        raise SystemExit(
            f"{args.adapter} 里没有 adapter_config.json —— 这不是一个 LoRA "
            f"checkpoint。全量 finetune 的 checkpoint 不需要合并,直接给 eval 用。"
        )
    if args.output.exists():
        raise SystemExit(f"输出目录已存在,拒绝覆盖:{args.output}")

    import torch  # noqa: F401  (让 dtype 解析走 torch)
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"加载底模 {args.base}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype="bfloat16", trust_remote_code=True
    )
    print(f"挂上 adapter {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    print("合并中(merge_and_unload)", flush=True)
    model = model.merge_and_unload()

    saved_modules = json.loads(adapter_cfg.read_text()).get("modules_to_save") or []
    if any(("lm_head" in m) or ("embed_tokens" in m) for m in saved_modules) and getattr(
        model.config, "tie_word_embeddings", False
    ):
        # modules_to_save 给 lm_head / embed_tokens 各建了独立副本,训练已解
        # tie。config 若仍写 tied,save_pretrained 只存一张矩阵,训好的
        # lm_head 会被静默丢掉、加载时再从 embed_tokens 重建 —— 权重就错了。
        model.config.tie_word_embeddings = False
        # 多模态 config 是嵌套的:save/load 两侧都可能查 text_config 里的
        # 同名 flag,只改顶层会在 from_pretrained 时被重新 tie,两张矩阵
        # 互相覆盖。两层一起改。
        text_cfg = getattr(model.config, "get_text_config", lambda: None)()
        if text_cfg is not None and getattr(text_cfg, "tie_word_embeddings", False):
            text_cfg.tie_word_embeddings = False
        print("底模 tied 且 adapter 训练了输出头副本:合并产物解 tie(含嵌套 text_config),分别保存两张矩阵", flush=True)

    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, safe_serialization=True)

    # processor / tokenizer 跟着底模走 —— LoRA 不改词表,但 eval 要在同一个
    # 目录里找到它们,否则加载会掉回去找 base 而把路径依赖藏起来。
    try:
        processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
        processor.save_pretrained(args.output)
    except Exception as error:  # noqa: BLE001
        print(f"processor 没能从底模保存({type(error).__name__}),逐文件兜底", flush=True)
        for name in (
            "chat_template.jinja",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ):
            src = args.base / name
            if src.is_file():
                shutil.copy2(src, args.output / name)

    provenance = {
        "merged_from_base": str(args.base),
        "merged_from_adapter": str(args.adapter),
        "adapter_config": json.loads(adapter_cfg.read_text()),
        "note": "这是 LoRA 合并产物,不是全量 finetune 的 checkpoint",
    }
    (args.output / "lora_merge_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False)
    )

    weights = sorted(p.name for p in args.output.glob("*.safetensors"))
    print(f"完成:{args.output}", flush=True)
    print(f"  权重文件:{weights}", flush=True)
    print("  provenance 写在 lora_merge_provenance.json", flush=True)


if __name__ == "__main__":
    main()
