"""训练时每个 optimizer step 算一次 MFU,连同 tokens/s、step 时长一起进 W&B(单独一张 mfu/ 表)。

MFU(Model FLOPs Utilization)= 模型"有用"的浮点运算速率 / 硬件峰值。有用功按经典算法:
每个 token 前向 2N、反向 4N,合计 6N(N 是参与训练的参数量),再加 full attention 层里随
序列长度平方长的那一项;gradient checkpointing 的重算、ZeRO 通信、数据加载都不算(算进去
的叫 HFU)。所以它回答的是"机器有多少比例在干模型的活",per-device batch 1 加 4 个
micro-batch 串行的时候这个数很低(2026-09-17 4B 在 s3 上离线算是 7% 左右),这是要不要做
micro-batch > 1 的硬证据。

每一步的有用 FLOPs 这样算(单位:FLOP):
    6 * N_text * tokens                     文本塔,fwd + bwd
  + 2 * N_vision * image_tokens             视觉塔冻结,只有 fwd
  + 6 * L_full * d_attn * sum(T_i^2)        full attention 层的 QK^T 与 AV:fwd 4·T²·d,bwd 2 倍,
                                            causal 只算下三角再减半 → 6·T²·d;每条样本 T_i 各算各的
Qwen3.5 是 hybrid:每 4 层一层 full attention(4B / 9B 是 32 层里 8 层,27B 是 64 层里 16 层),
其余是 linear attention,O(T),它们的参数已经在 N_text 里,不另加。d_attn = num_heads × head_dim
(4B / 9B 是 16 × 256,27B 是 24 × 256),GQA 不改这一项。

峰值是**按卡的型号 × 训练精度查表**的(精度从 TrainingArguments 的 bf16 / fp16 自动判,都没开
再看 TF32 开没开),查不到就不算 MFU(只记 tokens/s 与 step 时长),要用别的卡在 PEAK_TFLOPS 里
加一行或者设 MFU_PEAK_TFLOPS 环境变量覆盖。**这份表现在只对 L20D / B300 认真核过:**
robbys7 上 nvidia-smi 报的 "NVIDIA L20D" 其实是 B300 —— torch 报 compute capability (10, 3)、
148 个 SM、268 GiB,和 NVIDIA 给 B300 的规格一致 —— 所以按 name 查到 L20D 之后再看 cc:
(10, 3) 走 B300 的峰值,(8, 9) 才是真的 Ada 架构 L20。峰值取的是 dense(不带 2:4 sparsity)
的数字,因为训练里没有 sparsity;NVIDIA 的市场页面标的往往是 sparsity 数,要除以 2。
B300 那一行的出处:NVIDIA 只在 GB300 NVL72 页面给整机 sparsity 数,单卡 dense BF16 2.25 PF /
FP8 4.5 PF / FP4 15 PF / FP32 75 TF 是各家整理的数据表(runpod / spheron / server-parts.eu /
flopper.io,2026-09-17 查),与 B200 的 dense BF16 一致。
"""

from __future__ import annotations

import os
import time

import torch

# 卡型号 → 各精度的 dense 峰值(TFLOP/s,每卡)。**精度按训练实际用的自动选**(见 detect_precision):
# 我们的 recipe 是 bf16 混合精度,矩阵乘走 BF16 tensor core,查 "bf16";没开 bf16/fp16 的 fp32 训练,
# 矩阵乘要么走 TF32(torch.backends.cuda.matmul.allow_tf32)要么真 FP32,分别查 "tf32" / "fp32"。
# 数字一律取 dense(不带 2:4 sparsity):训练里没有 sparsity,NVIDIA 市场页标的往往是 sparsity 数,
# 要除以 2。只对 L20D / B300 那一行认真核过(来源见文件头),别的型号按 NVIDIA 数据表抄,用之前
# 自己再对一遍(尤其 H20 这种阉割卡,不同批次数字不一样)。
PEAK_TFLOPS = {
    # B300(Blackwell Ultra):BF16/FP16 2.25 PF、FP8 4.5 PF、FP4 15 PF、FP32 75 TF(dense)。
    # BF16 dense 与 B200 相同,Blackwell Ultra 提的是 FP4 与显存(288 GB HBM3e)。robbys7 上叫 "NVIDIA L20D"。
    "NVIDIA B300": {"bf16": 2250.0, "fp16": 2250.0, "fp8": 4500.0, "tf32": 1125.0, "fp32": 75.0},
    "NVIDIA B200": {"bf16": 2250.0, "fp16": 2250.0, "fp8": 4500.0, "tf32": 1125.0, "fp32": 75.0},
    "NVIDIA H100 80GB HBM3": {"bf16": 989.0, "fp16": 989.0, "fp8": 1979.0, "tf32": 495.0, "fp32": 67.0},  # SXM;PCIe 是 756
    "NVIDIA H800": {"bf16": 989.0, "fp16": 989.0, "fp8": 1979.0, "tf32": 495.0, "fp32": 67.0},
    "NVIDIA H20": {"bf16": 148.0, "fp16": 148.0, "fp8": 296.0, "tf32": 74.0, "fp32": 44.0},
    "NVIDIA L20": {"bf16": 119.5, "fp16": 119.5, "fp8": 239.0, "tf32": 59.8, "fp32": 59.8},   # 真 L20(Ada,cc 8.9)
    "NVIDIA A100-SXM4-80GB": {"bf16": 312.0, "fp16": 312.0, "tf32": 156.0, "fp32": 19.5},
}


def detect_precision(training_args=None, allow_tf32=None) -> str:
    """训练实际用的矩阵乘精度:bf16 / fp16 / tf32 / fp32。
    bf16 与 fp16 混合精度下 matmul 走对应的 tensor core;都没开就是 fp32 参数,这时看 TF32 开没开。
    FP8 这套代码没用(transformer-engine 那条路),所以这里不会返回 fp8。"""
    if training_args is not None:
        if getattr(training_args, "bf16", False):
            return "bf16"
        if getattr(training_args, "fp16", False):
            return "fp16"
    if allow_tf32 is None:
        allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else False
    return "tf32" if allow_tf32 else "fp32"


def peak_tflops_for(device_name: str, capability: tuple[int, int], precision: str = "bf16") -> tuple[float | None, str]:
    """返回 (峰值 TFLOP/s, 说明)。查不到返回 (None, 原因)。"""
    override = os.environ.get("MFU_PEAK_TFLOPS")
    if override:
        return float(override), f"MFU_PEAK_TFLOPS={override}(环境变量覆盖,精度 {precision})"
    name = device_name.strip()
    key, note = None, ""
    if name == "NVIDIA L20D":
        # robbys7 的 "L20D" 标签下面是 B300(cc 10.3,148 SM,268 GiB);真 L20 是 Ada cc 8.9。
        if capability[0] >= 10:
            key, note = "NVIDIA B300", "L20D 标签、cc %d.%d → 按 B300" % capability
        else:
            key, note = "NVIDIA L20", "L20D 标签、cc %d.%d → 按真 L20(Ada)" % capability
    else:
        for candidate in PEAK_TFLOPS:
            if name.startswith(candidate):
                key, note = candidate, f"{name} → 表里 {candidate}"
                break
    if key is None:
        return None, f"{name}(cc {capability[0]}.{capability[1]})不在 PEAK_TFLOPS 表里,只记 tokens/s,不算 MFU"
    value = PEAK_TFLOPS[key].get(precision)
    if value is None:
        return None, f"{note},但表里没有 {precision} 的峰值,只记 tokens/s,不算 MFU"
    return value, f"{note},{precision} dense {value:.0f} TFLOP/s"


class MfuMeter:
    """每个 optimizer step 的账簿。compute_loss 里 observe() 一次 micro-batch,
    on_step_end 时 finish_step() 出数。所有 rank 都要调 observe(里面有 gather)。"""

    def __init__(self, *, n_text, n_vision, n_full_attention_layers, d_attn, image_token_id,
                 world_size, gather, report_flops=True):
        self.report_flops = report_flops
        self.n_text = float(n_text)
        self.n_vision = float(n_vision)
        self.n_full = int(n_full_attention_layers)
        self.d_attn = float(d_attn)
        self.image_token_id = image_token_id
        self.world_size = int(world_size)
        self.gather = gather
        self.peak_tflops, self.peak_note = (None, "")
        self._reset_step()
        self.step_started = None

    def describe_device(self, precision="bf16"):
        self.precision = precision
        if precision is None:                    # 精度还没定,先不查表
            self.peak_tflops, self.peak_note = None, "精度未定"
            return
        if not torch.cuda.is_available():
            self.peak_note = "没有 CUDA,不算 MFU"
            return
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        self.peak_tflops, self.peak_note = peak_tflops_for(name, cap, precision)

    def _reset_step(self):
        self.tokens = 0.0
        self.image_tokens = 0.0
        self.seq_sq = 0.0
        self.micro_batches = 0

    def observe(self, inputs, seq_lens=None):
        """一个 micro-batch 的 token 数、图 token 数、每条样本 T²,跨 rank 求和后累进本步。

        packing 时一行里有 k 条 decision、没有 attention_mask,长度要从 collator 给的 _sft_seq_lens 来:
        attention 的 FLOPs 是每条各算各的 T²(varlen 就是按段做的),拿整行的 T² 会高估一大截。
        """
        ids = inputs["input_ids"]
        mask = inputs.get("attention_mask")
        if seq_lens is not None:
            lengths = seq_lens.reshape(-1).to(device=ids.device, dtype=torch.float64)
        elif mask is None:
            lengths = torch.full((ids.shape[0],), ids.shape[1], device=ids.device, dtype=torch.float64)
        else:
            lengths = mask.sum(dim=-1).to(torch.float64)
        image = (ids == self.image_token_id).sum().to(torch.float64) if self.image_token_id is not None \
            else torch.zeros((), device=ids.device, dtype=torch.float64)
        local = torch.stack([lengths.sum(), image, (lengths * lengths).sum()])
        total = self.gather(local).reshape(-1, 3).sum(0)
        self.tokens += float(total[0])
        self.image_tokens += float(total[1])
        self.seq_sq += float(total[2])
        self.micro_batches += 1

    def start_step(self):
        self.step_started = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._reset_step()

    def step_flops(self):
        return (6.0 * self.n_text * self.tokens
                + 2.0 * self.n_vision * self.image_tokens
                + 6.0 * self.n_full * self.d_attn * self.seq_sq)

    def finish_step(self):
        """返回这一步要记的指标;没开始过或没样本就返回空。"""
        if self.step_started is None or self.tokens == 0:
            return {}
        dt = max(time.time() - self.step_started, 1e-6)
        flops = self.step_flops()
        out = {
            "step_time_s": dt,
            "tokens_per_step": self.tokens,
            "tokens_per_sec": self.tokens / dt,
            # 峰值显存:packing 之后这是选 budget 的硬约束,顺手记进同一张表(每步重置,记的是这一步的峰值)
            "peak_memory_gb": (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0,
        }
        if self.report_flops:
            out["model_tflops_per_gpu"] = flops / dt / self.world_size / 1e12
        if self.report_flops and self.peak_tflops:
            out["mfu"] = out["model_tflops_per_gpu"] / self.peak_tflops
        return out


WANDB_SECTION = "mfu"   # W&B 里单开一张表:key 前缀决定分区,train/ 与 system/ 之外再来一个 mfu/


class MfuCallback:
    """挂在 Trainer 上:on_step_begin 计时,on_step_end 出数。要在 Trainer 建好之后把 .trainer 指过来。

    有 W&B 时直接 wandb.log 一组 mfu/* 的 key —— HF 的 WandbCallback 会把 Trainer 自己的 log 全套上
    train/ 前缀,走它那条路只能挤在 train 那张表里;这里用 commit=False 挂到同一 step 那一行上,
    Trainer 随后提交 loss 时一起落盘,x 轴仍是 train/global_step。只主进程发。
    没有 W&B 时退回塞进 TRL 的 _metrics,随 loss 那行打进 master.log。"""

    def __init__(self, meter: MfuMeter):
        self.meter = meter
        self.trainer = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.meter.start_step()
        return control

    def _wandb_run(self):
        try:
            import wandb
        except ImportError:
            return None
        return wandb.run

    def on_step_end(self, args, state, control, **kwargs):
        metrics = self.meter.finish_step()
        if not metrics or self.trainer is None:
            return control
        run = self._wandb_run()
        if run is not None:
            if self.trainer.is_world_process_zero():
                row = {f"{WANDB_SECTION}/{key}": float(value) for key, value in metrics.items()}
                row["train/global_step"] = int(state.global_step)
                run.log(row, commit=False)
        else:
            bucket = self.trainer._metrics["train" if self.trainer.model.training else "eval"]
            for key, value in metrics.items():
                bucket[key].append(float(value))
        return control

    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda *a, **k: k.get("control")
        raise AttributeError(name)
