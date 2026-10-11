#!/usr/bin/env python3
"""VLM SFT entry point for exported successful and recovery trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText, TrainingArguments
from trl import SFTConfig, SFTTrainer

from data_pipeline.manifest import check_split_groups, load_manifest, sha256
from training.recipe import build_run_record, parse_recipe, record_run, validate_resume_run
from training.sampling import EpisodeSampler, LengthGroupedSampler
from training.lora_support import lora_vision_exclusions
from training.model_loading import parameter_numel, validate_zero3_init
from training.jsonl_input import load_jsonl_input
from training.qwen35_preflight import (
    load_verified_processor,
    verify_attention_backend,
    verify_training_stack,
)
from training.history_truncation import (
    add_length_arguments,
    apply_length_policy_plan,
    prepare_length_policy_distributed,
)
from training.vlm_collator import VlmCollator
from training.oss_checkpoint import make_callback as make_oss_callback
from training.span_metrics import SpanMetrics, SpanMetricsCallback, StepSpanMeter
from training.mfu import MfuCallback, MfuMeter
from training.task_metrics import TaskMeter, TaskMetricsCallback
from training.wandb_logging import wandb_session


class FiniteSFTTrainer(SFTTrainer):
    """Stop every rank before backward when any rank has a non-finite loss."""

    def __init__(self, *args, sampling_strategy="turn_uniform", samples_per_epoch=None,
                 max_sample_repeats=None, loss_normalization="turn_mean", row_lengths=None,
                 diag_param_scan=False, mfu_meter=None, task_meter=None, span_meter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mfu_meter = mfu_meter
        self.task_meter = task_meter
        self.span_meter = span_meter
        if self.args.per_device_train_batch_size != 1 or self.args.per_device_eval_batch_size != 1:
            raise ValueError("VLM completion masking requires per-device batch size 1")
        # loss 归一两种口径,由 recipe 的 loss_normalization 决定:
        #   turn_mean(默认):model_accepts_loss_kwargs=False,Trainer 不传 num_items_in_batch,
        #     TRL 的 chunked_nll 对每条 decision 除自己的有效 token 数,再跨 rank / 梯度累积取均值,
        #     每条 decision 等权。这就是老入口(HF 对 Qwen3.5 自动探测出 False)的行为:c910、
        #     20260911_20646rows(record 337167030)与 lcy 分支 9B 那次都是它 —— 09-12 合并时用
        #     training/tools/dump_first_batch.py 在老入口里实测 model_accepts_loss_kwargs=False、
        #     num_items_in_batch=None 证实。
        #   token_mean:打开 model_accepts_loss_kwargs,Trainer 把全局 batch 的有效 token 数传给
        #     loss,sum / 全局有效 token 数,长 completion 权重大。没有任何现役 run 用过,只作 opt-in。
        if loss_normalization not in {"token_mean", "turn_mean"}:
            raise ValueError(f"unknown loss_normalization: {loss_normalization}")
        self.loss_normalization = loss_normalization
        self.model_accepts_loss_kwargs = loss_normalization == "token_mean"
        self._diag_row_log = bool(diag_param_scan)
        self.episode_sampler = None
        if sampling_strategy == "length_grouped":
            if row_lengths is None or len(row_lengths) != len(self.train_dataset):
                raise ValueError("length_grouped sampling needs one measured length per training row")
            self.episode_sampler = LengthGroupedSampler(row_lengths, seed=self.args.data_seed)
        elif sampling_strategy != "turn_uniform":
            self.episode_sampler = EpisodeSampler(
                self.train_dataset, strategy=sampling_strategy, seed=self.args.data_seed,
                num_samples=samples_per_epoch, max_repeats=max_sample_repeats,
            )
        elif samples_per_epoch is not None or max_sample_repeats is not None:
            raise ValueError("Sampling draw/repeat limits require episode sampling")

    def _get_train_sampler(self, train_dataset=None):
        return self.episode_sampler if self.episode_sampler is not None else super()._get_train_sampler(train_dataset)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        row_index = inputs.pop("_sft_row_index", None)
        span = inputs.pop("_sft_span", None)
        meter = getattr(self, "mfu_meter", None)
        if meter is not None:
            meter.observe(inputs)
        if getattr(self, "_diag_row_log", False):
            _r = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            _row = (int(row_index.reshape(-1)[0].item())
                    if isinstance(row_index, torch.Tensor) and row_index.numel() else row_index)
            print(f"ROW_AT_STEP rank={_r} global_step={int(self.state.global_step)} "
                  f"row_index={_row}", flush=True)
        labels = inputs.get("labels")
        sequence_length = int(labels.shape[-1]) if labels is not None else -1
        supervised_tokens = (
            int((labels[..., 1:] != -100).sum().item())
            if labels is not None and labels.shape[-1] > 1
            else 0
        )
        has_supervision = torch.tensor(int(supervised_tokens > 0), device=self.accelerator.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(has_supervision, op=dist.ReduceOp.MIN)
        if not bool(has_supervision.item()):
            raise ValueError(
                "at least one rank's SFT batch has no next-token supervision: "
                f"row_index={row_index}, sequence_length={sequence_length}"
            )

        # reasoning / tool_call 两段各自的 accuracy 与 loss:搭在 TRL 的 chunked loss 上按
        # span 分账,不另跑 lm_head、不另做 all-gather。见 training/span_metrics.py。
        with SpanMetrics.collect(span, labels) as spans:
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        self._last_spans = spans     # 给测试与诊断看账簿
        task_meter = getattr(self, "task_meter", None)
        if task_meter is not None and spans is not None:
            task_meter.observe(spans, row_index)   # 按 task 归账,step 结束时 callback 一起 gather
        if spans is not None:
            span_meter = getattr(self, "span_meter", None)
            if span_meter is not None and self.model.training:
                # TRL 自己那条 mean_token_accuracy 是"每个 micro-batch 先算比值、再对 accum 个比值等权
                # 平均",和我们统一到 loss_normalization 的口径对不上。它每个 micro-batch 都会往
                # _metrics 里塞一个,这里随手清掉,step 结束时由 SpanMetricsCallback 用同一个名字塞回
                # 我们自己算的那个数(用户拍板:曲线名不变,值换成对的)。
                self._metrics["train"].pop("mean_token_accuracy", None)
                # 训练时只记账,整个 optimizer step 结束再算一次(SpanMetricsCallback)。
                # 以前是每个 micro-step 算一个比值塞进 _metrics、TRL 再把这些比值等权平均,
                # 那样 accum > 1 时是"比值的平均"、跨 rank 又是 token 加权,和 train/loss 的
                # turn_mean 口径对不上(2026-09-17 单 task run 上 reasoning_loss 差 0.8%)。
                span_meter.observe(spans)
            else:
                # eval 没有 accum,老路径照旧:每个 micro-batch 一个数。用 accelerator.gather 不用
                # gather_for_metrics —— 后者在 epoch 末按剩余样本数截断,计数向量会 reshape 不回去。
                mode = "train" if self.model.training else "eval"
                for key, value in spans.reduce(self.accelerator.gather).items():
                    self._metrics[mode][key].append(value)
        if os.environ.get("PRINT_PEAK_MEM") == "1" and torch.cuda.is_available():
            print(json.dumps({
                "event": "peak_memory", "global_step": int(self.state.global_step),
                "sequence_length": sequence_length, "supervised_tokens": supervised_tokens,
                "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1),
            }, sort_keys=True), flush=True)
        local_finite = bool(torch.isfinite(loss.detach()).all().item())
        all_finite = torch.tensor(
            1 if local_finite else 0,
            device=loss.device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(all_finite, op=dist.ReduceOp.MIN)
        if not bool(all_finite.item()):
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            local_row = (
                int(row_index.reshape(-1)[0].item())
                if isinstance(row_index, torch.Tensor) and row_index.numel()
                else row_index
            )
            diagnostic = {
                "event": "non_finite_sft_loss",
                "rank": rank,
                "local_loss": float(loss.detach().float().item()),
                "local_loss_finite": local_finite,
                "row_index": local_row,
                "sequence_length": sequence_length,
                "supervised_tokens": supervised_tokens,
            }
            print(json.dumps(diagnostic, sort_keys=True), file=sys.stderr, flush=True)
            raise FloatingPointError(
                "non-finite SFT loss detected across distributed ranks; "
                "refusing backward and optimizer update"
            )
        return (loss, outputs) if return_outputs else loss


def restore_token_counter(trainer, checkpoint):
    """续训时把 TRL 的 num_tokens 计数接上。

    TRL 的 SFTTrainer._total_train_tokens 是进程内的累计(init 置 0、compute_loss 累加、log 写出),
    不在 trainer_state.json 里,resume 后从 0 重数,W&B 上 train/num_tokens 在续训点断崖
    (2026-09-17 4B 那轮:68 步 7.71e7 → 69 步 2.39e6)。HF 存 checkpoint 时 trainer_state.json 的
    log_history 是写到那一步的,最后一条带 num_tokens 的就是续训点的累计,拿它续上。
    找不到就打一行,不拦训练。"""
    state_file = Path(checkpoint) / "trainer_state.json"
    try:
        history = json.loads(state_file.read_text()).get("log_history") or []
    except (OSError, ValueError) as exc:
        print(f"[resume] 读不到 {state_file}({type(exc).__name__}),num_tokens 从 0 数", flush=True)
        return None
    for entry in reversed(history):
        if "num_tokens" in entry:
            total = int(round(float(entry["num_tokens"])))
            trainer._total_train_tokens = total
            print(f"[resume] num_tokens 接上:{total}(checkpoint step {entry.get('step')})", flush=True)
            return total
    print("[resume] trainer_state.json 的 log_history 里没有 num_tokens,从 0 数", flush=True)
    return None


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--resume-with-universal-checkpoint",
        action="store_true",
        help=(
            "Resume a DeepSpeed universal checkpoint when changing WORLD_SIZE. "
            "This is a runtime-only opt-in; all recorded data and recipe fields "
            "other than world size must still match."
        ),
    )
    parser.add_argument("--sampling-strategy", choices=("turn_uniform", "episode_uniform", "task_episode_uniform", "length_grouped"), default="turn_uniform",
                        help="length_grouped:按 length preflight 的精确 token 数分组,同一步样本长度相近(4 node 27B 上 100 s/it → 27 s/it),每 epoch 每行一次")
    parser.add_argument("--loss-normalization", choices=("turn_mean", "token_mean"), default="turn_mean",
                        help="turn_mean = 每条 decision 等权(老入口与所有现役 run 的实际口径,见 FiniteSFTTrainer 注释);token_mean = 全局 batch 的 token 均值(opt-in,没有现役 run 用过)")
    parser.add_argument("--diag-param-scan", action="store_true",
                        help="诊断用(默认关):每个 optimizer step 后扫参数有限性并逐步打印各 rank 的 row_index,定位 NaN")
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--max-sample-repeats", type=int)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-strategy", choices=("steps", "epoch"), default="steps")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    add_length_arguments(parser)
    parser.add_argument(
        "--freeze-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    # ---- LoRA(默认关。开着才走 peft,关着的代码路径与全量 finetune 逐字节相同)
    parser.add_argument(
        "--lora",
        action=argparse.BooleanOptionalAction, default=False,
        help="用 LoRA 代替全量 finetune。默认关 —— 已有的 launcher、"
             "checkpoint 口径、eval 通路都不受影响。开了之后 trainer 存的是"
             "adapter(adapter_model.safetensors),不是 model.safetensors,"
             "要给 eval 用得先经 training/merge_lora_checkpoint.py 合并。",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument(
        "--lora-alpha", type=int, default=64,
        help="惯例是 rank 的两倍;缩放系数 alpha/rank 决定 adapter 的有效步长",
    )
    # 每存一次 checkpoint 就传 OSS。不给 --oss-bucket 就完全不启用;缺 config 或缺 oss2
    # 也只打一行说明然后继续训,不会把 job 判失败。见 training/oss_checkpoint.py。
    parser.add_argument("--oss-bucket", default=None,
                        help="传到哪个 bucket,例 antsys-robbys3-prod。不给就不传")
    parser.add_argument("--oss-root", default=None,
                        help="bucket 里的前缀,默认取 --output 的目录名")
    parser.add_argument("--oss-config", default=None,
                        help="ossutil 格式的 config 路径,默认 ~/.ossutilconfig。"
                             "凭据只从这个文件读,不进日志也不进 run record")
    parser.add_argument("--oss-delete-local", action=argparse.BooleanOptionalAction, default=False,
                        help="全部传成功之后删掉本地那份(照 hatssl 的做法);默认不删")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="注入 LoRA 的模块名(按后缀匹配)。不用 all-linear:那会把 vision "
             "塔也注进去,与 --freeze-vision 的意图相反。",
    )
    parser.add_argument(
        "--lora-modules-to-save",
        default="",
        help="逗号分隔的模块名,不注 LoRA 而是整个全量训练并存进 adapter"
             "(peft modules_to_save)。例 lm_head,embed_tokens:对着"
             "\"不发 tool call\"的失败形态把输出头放开训。注意底模"
             " tie_word_embeddings=true 时,peft 会给二者各建独立副本"
             "(训练即解 tie);merge_lora_checkpoint.py 会把合并产物的"
             " config 改成不 tie,否则 save_pretrained 只存一张矩阵,"
             "训好的 lm_head 会被静默丢掉。",
    )
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--zero3-init", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
        help="Opt into partitioned model loading; requires a ZeRO-3 config. "
             "Omit to preserve the established 9B loading and resume path.",
    )
    parser.add_argument(
        "--flashnext-qsa", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
        help="Experimental Qwen4Exp-only Miles sparse attention adapter; requires --zero3-init.",
    )
    parser.add_argument(
        "--deepspeed",
        type=lambda value: None if value.lower() == "none" else value,
        default=None,
    )
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument("--run-name")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-strategy", choices=("steps", "epoch"), default="steps")
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Trade compute for activation memory. Disable on high-memory GPUs "
            "when the full activation footprint fits."
        ),
    )
    parser.add_argument(
        "--hard-exit-after-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After saving the final model and metrics, destroy the distributed "
            "process group and use os._exit(0). This narrowly avoids an observed "
            "native-library SIGSEGV during Python interpreter teardown."
        ),
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help=(
            "Collate rows (image decode + multimodal chat rendering) in this "
            "many prefetching DataLoader workers instead of the training "
            "thread. Zero preserves the historical in-thread behavior."
        ),
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per worker; used only when workers > 0.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Bit-level reproducible training: closes the four verified GPU "
            "nondeterminism sources (FA2 backward atomics, torch-level "
            "atomics, fla triton autotune selection, causal_conv1d backward "
            "atomics). Same recipe on the same environment family then "
            "reproduces checkpoints hash-identically, provided the launcher "
            "also uses a cold or frozen JIT kernel cache -- see "
            "docs/DETERMINISTIC_TRAINING.md."
        ),
    )
    parser.add_argument(
        "--fla-config-dir",
        default=None,
        help=(
            "With --deterministic: directory of dumped fla kernel configs "
            "(FLA_CONFIG_DIR). Replaces the conservative autotune pin with "
            "fixed benchmark-selected configs, recovering most of the "
            "deterministic-mode step-time overhead."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parse_recipe(parser)
    if args.resume_with_universal_checkpoint and not args.resume_from_checkpoint:
        parser.error("--resume-with-universal-checkpoint requires --resume-from-checkpoint")
    deepspeed_config = args.deepspeed
    if args.resume_with_universal_checkpoint:
        if args.deepspeed:
            deepspeed_config = json.loads(Path(args.deepspeed).read_text())
            checkpoint_config = deepspeed_config.setdefault("checkpoint", {})
            if not isinstance(checkpoint_config, dict):
                parser.error("DeepSpeed checkpoint config must be a JSON object")
            checkpoint_config["load_universal"] = True
        else:
            parser.error("--resume-with-universal-checkpoint requires --deepspeed")
        from training.deepspeed_universal import enable_grouped_zero3_universal_load

        enable_grouped_zero3_universal_load()
    if args.max_input_tokens <= 0:
        parser.error("--max-input-tokens must be positive")
    if args.global_batch_size <= 0 or args.eval_steps <= 0:
        parser.error("batch size and eval steps must be positive")
    if args.sampling_strategy == "turn_uniform" and (args.samples_per_epoch is not None or args.max_sample_repeats is not None):
        parser.error("Sampling draw/repeat limits require episode sampling")
    if args.dataloader_num_workers < 0 or args.dataloader_prefetch_factor < 1:
        raise ValueError(
            "dataloader workers must be >= 0 and prefetch factor >= 1"
        )
    if args.fla_config_dir and not args.deterministic:
        raise ValueError("--fla-config-dir requires --deterministic")
    if args.deterministic:
        from training.deterministic import enable_deterministic_training

        applied = enable_deterministic_training(args.fla_config_dir)
        print(f"[deterministic] enabled: {', '.join(applied)}")
    if getattr(args, "zero3_init", False):
        validate_zero3_init(deepspeed_config)
    if getattr(args, "flashnext_qsa", False):
        if not getattr(args, "zero3_init", False) or args.attn_implementation != "sdpa" or args.lora:
            parser.error("--flashnext-qsa requires --zero3-init, --attn-implementation sdpa and full SFT")
        from transformers import AutoConfig
        if AutoConfig.from_pretrained(args.model).model_type != "qwen4_exp":
            parser.error("--flashnext-qsa is only for Qwen3.8-Flash-Next (qwen4_exp)")
        from training.backends.flashnext.attention import install
        install()
    dataset_root, splits = load_manifest(args.data_manifest, args.data_root)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.gradient_accumulation_steps is None:
        if args.global_batch_size % world_size:
            raise ValueError(
                "global batch size must be divisible by WORLD_SIZE because "
                "per-device batch size is fixed at one"
            )
        gradient_accumulation_steps = args.global_batch_size // world_size
    else:
        gradient_accumulation_steps = args.gradient_accumulation_steps
        resolved_global_batch = world_size * gradient_accumulation_steps
        if resolved_global_batch != args.global_batch_size:
            raise ValueError(
                "global batch mismatch: WORLD_SIZE * gradient accumulation "
                f"= {resolved_global_batch}, requested {args.global_batch_size}"
            )
    dataset, estimated_context_tokens = load_jsonl_input(splits["train"])
    validation_dataset = None
    if args.validation:
        if not splits["validation"]:
            raise ValueError("Validation enabled but manifest has no validation files")
        validation_dataset, _ = load_jsonl_input(splits["validation"], row_index_offset=len(dataset))
        check_split_groups(dataset, validation_dataset)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        count = len(estimated_context_tokens)
        print(
            json.dumps(
                {
                    "event": "sft_context_length_preflight",
                    "rows": count,
                    "estimated_tokens_min": estimated_context_tokens[0],
                    "estimated_tokens_p50": estimated_context_tokens[count // 2],
                    "estimated_tokens_p90": estimated_context_tokens[int(0.9 * (count - 1))],
                    "estimated_tokens_p99": estimated_context_tokens[int(0.99 * (count - 1))],
                    "estimated_tokens_max": estimated_context_tokens[-1],
                    "exact_rendered_token_limit": args.max_input_tokens,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    # Run this before allocating model weights. A too-old Transformers build can
    # silently resolve the VLM checkpoint to a tokenizer-only AutoProcessor.
    verify_training_stack()
    processor = load_verified_processor(args.model)
    verify_attention_backend(args.attn_implementation)
    tokenizer = processor.tokenizer
    for name in ("pad_token", "eos_token", "pad_token_id", "eos_token_id"):
        if not hasattr(processor, name):
            setattr(processor, name, getattr(tokenizer, name))
    if not hasattr(processor, "convert_tokens_to_ids"):
        processor.convert_tokens_to_ids = tokenizer.convert_tokens_to_ids

    base_evidence = dict(
        manifest_sha256=sha256(args.data_manifest),
        recipe_sha256=sha256(args.recipe),
        world_size=world_size,
        loss={"type": "chunked_nll", "normalization": args.loss_normalization, "supervised_token_weight": 1.0},
        sampling={"strategy": args.sampling_strategy},
        processor_class=type(processor).__name__,
        model_files_sha256={name: sha256(Path(args.model) / name) for name in (
            "config.json", "processor_config.json", "preprocessor_config.json",
            "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
        ) if (Path(args.model) / name).is_file()},
    )
    validate_resume_run(args, base_evidence)
    cache_identity = hashlib.sha256(json.dumps(
        build_run_record(args, base_evidence), sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    cache_path = Path(args.output) / "length_policy_plan.json"
    cached = None
    if cache_path.is_file():
        candidate = json.loads(cache_path.read_text())
        if candidate.get("version") == 1 and candidate.get("identity_sha256") == cache_identity:
            cached = candidate

    if cached is not None:
        dataset, history_cuts, length_stats = apply_length_policy_plan(
            dataset, cached["train"], args.max_input_tokens, args.truncate_history,
        )
        validation_stats = None
        if validation_dataset is not None:
            validation_dataset, validation_cuts, validation_stats = apply_length_policy_plan(
                validation_dataset, cached["validation"],
                args.max_input_tokens, args.truncate_history,
            )
            history_cuts.update(validation_cuts)
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(json.dumps({"event": "sft_length_policy_cache_hit",
                              "path": str(cache_path)}, sort_keys=True), flush=True)
    else:
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(json.dumps({"event": "sft_length_policy_start", "rows": len(dataset),
                              "world_size": world_size}, sort_keys=True), flush=True)
        dataset, history_cuts, length_stats, train_plan = prepare_length_policy_distributed(
            dataset, processor, dataset_root, args.max_input_tokens, args.truncate_history,
        )
        validation_stats = None
        validation_plan = None
        if validation_dataset is not None:
            validation_dataset, validation_cuts, validation_stats, validation_plan = (
                prepare_length_policy_distributed(
                    validation_dataset, processor, dataset_root,
                    args.max_input_tokens, args.truncate_history,
                )
            )
            history_cuts.update(validation_cuts)

    row_lengths = None
    if args.sampling_strategy == "length_grouped":
        plan_for_lengths = cached["train"] if cached is not None else train_plan
        row_lengths = list(plan_for_lengths.get("tokens") or [])
        if len(row_lengths) != len(dataset) or any(x < 0 for x in row_lengths):
            raise ValueError(
                "length_grouped needs measured lengths for every retained row; "
                "delete the stale length_policy_plan.json cache and rerun the preflight"
            )
        LengthGroupedSampler(row_lengths, seed=args.seed)  # validate before allocating weights
    elif args.sampling_strategy != "turn_uniform":
        # Validate counts/identities before allocating model weights.
        EpisodeSampler(dataset, strategy=args.sampling_strategy, seed=args.seed,
                       num_samples=args.samples_per_epoch, max_repeats=args.max_sample_repeats)
    record_run(args, dict(
        **base_evidence,
        length_policy={"train": length_stats, "validation": validation_stats},
    ))
    if cached is None and int(os.environ.get("RANK", "0")) == 0:
        cache = {
            "version": 1,
            "identity_sha256": cache_identity,
            "train": train_plan,
            "validation": validation_plan,
        }
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cache, sort_keys=True) + "\n")
        temporary.replace(cache_path)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(json.dumps({"event": "sft_length_policy", **length_stats}, sort_keys=True), flush=True)
        if validation_stats is not None:
            print(json.dumps({"event": "sft_validation_length_policy", **validation_stats}, sort_keys=True), flush=True)

    def make_config():
        # Transformers 5.16 folds fractional warmup into warmup_steps. Preserve
        # the old API on the pinned 9B runtime, including its resume records.
        if "warmup_ratio" in TrainingArguments.__dataclass_fields__:
            warmup = {"warmup_ratio": args.warmup_ratio}
        else:
            if not 0 <= args.warmup_ratio < 1:
                raise ValueError("This Transformers runtime requires 0 <= warmup_ratio < 1")
            warmup = {"warmup_steps": args.warmup_ratio}
        return SFTConfig(
            output_dir=args.output,
            learning_rate=args.learning_rate,
            **warmup,
            max_grad_norm=args.max_grad_norm,
            lr_scheduler_type="linear",
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            eval_strategy=args.eval_strategy if args.validation else "no",
            eval_steps=args.eval_steps,
            prediction_loss_only=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            bf16=True,
            gradient_checkpointing=args.gradient_checkpointing,
            max_length=None,
            # The custom collator supplies precise completion-only labels because
            # Qwen's chat template does not expose TRL generation masks.
            assistant_only_loss=False,
            loss_type="chunked_nll",
            pad_token=tokenizer.pad_token,
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            report_to=args.report_to,
            run_name=args.run_name,
            # Never let Transformers replace NaN/Inf with a plausible-looking zero
            # in the log stream. FiniteSFTTrainer fails all ranks before backward.
            logging_nan_inf_filter=False,
            seed=args.seed,
            data_seed=args.seed,
            remove_unused_columns=False,
            # The custom VLM collator performs chat rendering, image processing,
            # and completion masking. TRL's text-only preparation expects a
            # `text` column and must not tokenize the opaque row_json column.
            dataset_kwargs={"skip_prepare_dataset": True},
            deepspeed=deepspeed_config,
            # Collate cost here is real work: ten PNG decodes plus a 40k-token
            # multimodal render per row. Workers overlap that CPU time with the
            # GPU step; zero keeps the exact historical single-thread path.
            dataloader_num_workers=args.dataloader_num_workers,
            **(
                {
                    "dataloader_persistent_workers": True,
                    "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
                }
                if args.dataloader_num_workers > 0
                else {}
            ),
        )

    # Initialize ZeRO before loading only when explicitly requested. Keep the
    # config alive: Transformers uses its weak reference during from_pretrained.
    config = make_config() if getattr(args, "zero3_init", False) else None

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    if getattr(args, "flashnext_qsa", False):
        model.config.use_cache = False
        model.config.text_config.use_cache = False
    total_parameters = sum(parameter_numel(parameter) for parameter in model.parameters())
    frozen = 0
    if args.freeze_vision:
        for name, parameter in model.named_parameters():
            if any(token in name.lower() for token in ("vision", "visual", "aligner")):
                parameter.requires_grad = False
                frozen += parameter_numel(parameter)
        if frozen == 0:
            raise ValueError(
                "--freeze-vision matched no vision/visual/aligner parameters"
            )
    lora_summary: dict[str, object] = {}
    if args.lora:
        from peft import LoraConfig, get_peft_model

        targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
        if not targets:
            raise ValueError("--lora-target-modules 不能为空")
        modules_to_save = [
            m.strip() for m in args.lora_modules_to_save.split(",") if m.strip()
        ]
        exclude: list[str] = []
        if args.freeze_vision:
            exclude = lora_vision_exclusions(
                (name for name, _ in model.named_modules()), targets
            )
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            modules_to_save=modules_to_save or None,
            exclude_modules=exclude or None,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Initialize adapters with the same configured seed on every rank,
        # before Trainer seeds its training RNG. Full finetuning adds no weights.
        from transformers import set_seed as _hf_set_seed

        _hf_set_seed(args.seed)
        model = get_peft_model(model, lora_config)
        # LoRA + gradient checkpointing 的经典坑:base 权重全部 frozen,
        # checkpoint 段重算时输入张量没有 grad,反向会炸在
        # "element 0 of tensors does not require grad"。让 embedding 的输出
        # 带上 requires_grad 就通了。全量 finetune 不需要这一步(base 本来就
        # 在训),所以只在 LoRA 分支里做。
        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        lora_summary = {
            "lora": True,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": targets,
            "lora_modules_to_save": modules_to_save,
            "lora_excluded_vision_modules": len(exclude),
            "lora_input_require_grads": bool(args.gradient_checkpointing),
        }

    trainable_parameters = sum(
        parameter_numel(parameter) for parameter in model.parameters() if parameter.requires_grad
    )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(
            json.dumps(
                {
                    "total_parameters": total_parameters,
                    "frozen_vision_parameters": frozen,
                    "trainable_parameters": trainable_parameters,
                    "trainable_percent": round(
                        100.0 * trainable_parameters / total_parameters, 4
                    ),
                    **lora_summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if config is None:
        config = make_config()
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"Qwen3.5 SFTConfig passed (sampling={args.sampling_strategy}, "
              f"loss_normalization={args.loss_normalization})", flush=True)

    collate_vlm = VlmCollator(
        processor, dataset_root, args.max_input_tokens,
        history_cuts=history_cuts,
    )
    diag_callbacks = []
    if args.diag_param_scan:
        from transformers import TrainerCallback

        class ParamScanCallback(TrainerCallback):
            """每个 optimizer step 后扫参数有限性 —— NaN 写入发生的步与层一次定位。"""

            def on_step_end(self, targs, state, control, model=None, **kw):
                bad = []
                for n, p in model.named_parameters():
                    if p.data.is_floating_point() and not torch.isfinite(p.data).all():
                        bad.append(n)
                        if len(bad) >= 6:
                            break
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                if bad:
                    print(f"PARAM_POISONED step={state.global_step} rank={rank} "
                          f"n>={len(bad)} first={bad}", flush=True)
                elif rank == 0:
                    print(f"PARAM_CLEAN step={state.global_step}", flush=True)

        diag_callbacks.append(ParamScanCallback())

    oss_callback = make_oss_callback(args)
    oss_callbacks = [oss_callback] if oss_callback is not None else []

    # MFU:每个 optimizer step 一个数,和 loss 一起进 W&B。峰值按卡型号查表(training/mfu.py),
    # 查不到只记 tokens/s。参数:文本塔可训练参数 6N,冻结视觉塔 2N,full attention 层的 T² 项。
    text_config = getattr(model.config, "text_config", model.config)
    layer_types = getattr(text_config, "layer_types", None) or []
    n_full = layer_types.count("full_attention") if layer_types else int(
        getattr(text_config, "num_hidden_layers", 0) // max(int(getattr(text_config, "full_attention_interval", 1) or 1), 1))
    d_attn = int(getattr(text_config, "num_attention_heads", 0)) * int(
        getattr(text_config, "head_dim", 0) or (getattr(text_config, "hidden_size", 0) // max(int(getattr(text_config, "num_attention_heads", 1)), 1)))
    mfu_meter = MfuMeter(
        n_text=trainable_parameters, n_vision=frozen, n_full_attention_layers=n_full, d_attn=d_attn,
        image_token_id=getattr(model.config, "image_token_id", None), world_size=world_size,
        # Dense 6N + full-attention FLOPs are not valid for sparse MoE/PLE/QSA.
        report_flops=not getattr(args, "flashnext_qsa", False),
        gather=lambda t: t)   # gather 在 trainer 建好后换成 accelerator 的
    mfu_meter.describe_device(precision=None)   # 精度等 SFTConfig 建好再判(bf16 / fp16 从它读)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(json.dumps({"event": "mfu_setup", "n_text": trainable_parameters, "n_vision": frozen,
                          "n_full_attention_layers": n_full, "d_attn": d_attn},
                         ensure_ascii=False), flush=True)
    mfu_callback = MfuCallback(mfu_meter)
    # 每个 task 六条曲线(W&B 的 task/ 表)。task 列表与每行的 task 都从 manifest 来;每一步就是
    # 这一步的原始数,不做滑动平均,loss 和 train/loss 一样是 turn_mean(每条 decision 等权)。
    manifest_rows = splits["train"].rows
    task_list = sorted({row["task_id"] for row in manifest_rows})
    task_meter = TaskMeter(task_list, [row["task_id"] for row in manifest_rows],
                           normalization=args.loss_normalization)
    task_callback = TaskMetricsCallback(task_meter)
    # train/ 里那几条分段曲线:和 task/ 表、和 train/loss 用同一套平均方式(loss_normalization)。
    span_meter = StepSpanMeter(normalization=args.loss_normalization)
    span_callback = SpanMetricsCallback(span_meter)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(json.dumps({"event": "task_metrics_setup", "tasks": task_list}, ensure_ascii=False), flush=True)

    trainer = FiniteSFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        eval_dataset=validation_dataset,
        sampling_strategy=args.sampling_strategy,
        samples_per_epoch=args.samples_per_epoch,
        max_sample_repeats=args.max_sample_repeats,
        loss_normalization=args.loss_normalization,
        row_lengths=row_lengths,
        diag_param_scan=args.diag_param_scan,
        processing_class=processor,
        data_collator=collate_vlm,
        callbacks=diag_callbacks + oss_callbacks + [mfu_callback, task_callback, span_callback],
        mfu_meter=mfu_meter,
        task_meter=task_meter,
        span_meter=span_meter,
    )
    mfu_meter.gather = trainer.accelerator.gather          # 不用 gather_for_metrics,见 compute_loss 里的注释
    mfu_callback.trainer = trainer
    task_callback.trainer = trainer
    span_callback.trainer = trainer
    from training.mfu import detect_precision
    mfu_meter.describe_device(precision=detect_precision(trainer.args))
    if trainer.is_world_process_zero():
        print(json.dumps({"event": "mfu_peak", "precision": mfu_meter.precision,
                          "peak_tflops_per_gpu": mfu_meter.peak_tflops, "note": mfu_meter.peak_note},
                         ensure_ascii=False), flush=True)
    if args.resume_from_checkpoint:
        restore_token_counter(trainer, args.resume_from_checkpoint)
    with wandb_session(args, is_main_process=trainer.is_world_process_zero()):
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        # Materialize the final model and metrics before the teardown workaround.
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if args.validation:
            trainer.save_metrics("eval", trainer.evaluate())

    if args.hard_exit_after_train:
        # On the pinned Hopper runtime, all ranks have completed computation and
        # saved successfully but native destructors can SIGSEGV during Python's
        # atexit phase. Explicitly close NCCL, flush logs, then bypass only that
        # unsafe destructor phase. Exceptions before this point still fail loud.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
