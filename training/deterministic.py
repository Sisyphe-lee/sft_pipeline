"""Qwen3.5 SFT 栈的确定性训练开关(bit 级可复现)。

2026-08-12/13 在 PickPlaceCan 上定位并验证的四个 GPU 级非确定源
(证据链见 docs/probe_determ_results.md):

1. flash_attention_2 backward 的 atomic add
   → FLASH_ATTENTION_DETERMINISTIC=1(transformers>=5.14 逐调用读取)
2. torch 级 atomics(embedding backward scatter_add 等)
   → use_deterministic_algorithms(warn_only) + cudnn.deterministic
   + CUBLAS_WORKSPACE_CONFIG
3. fla triton kernel 的 autotune 现场 benchmark 选型(timing 依赖)
   → Autotuner._bench 返回常数,min() 稳定取第一个 pruned config
4. causal_conv1d CUDA 包 backward 的 dweight/dbias 非确定
   → 整个替换成 torch F.conv1d 等价实现(与历史 convfix_wrapper 逐行
   同款),backward 走 cudnn,在 deterministic 模式下确定。
   曾改成"调用前 .contiguous() 保留 fused kernel"(快 17.4%),但那是
   另一个 realization:2026-08-26 双集群全量对照,闭环 SR 86→78 /
   92→78,contract failure 都翻三倍,已回退。要动这一处必须重跑双集群
   train+eval 才有决策资格,证据见
   docs/DETERMINISTIC_TRAINING.md。

此外 bit 级跨 launch 场景一致还要求**统一冷编译或固化的 kernel
cache**(JIT cache 世代也是自由度),那部分属于启动脚本职责;以及
**任何新初始化必须发生在 seed 之后**(第 5 源:LoRA adapter 初始化曾
早于 Trainer 的 set_seed,已在 train_sft.py 的 LoRA 分支内前置
set_seed(3407) 修复)。见
docs/DETERMINISTIC_TRAINING.md。

已验证:b300/sm_103 环境族内同 node、跨 node、容器↔pod 全量
10-epoch 逐 epoch hash 一致。

**正式训练只传 `--deterministic`,不要传 `fla_config_dir`。** 它能省掉
第 3 条保守选型的 perf 代价,但换选型 = 换 numerical realization:同数据
同超参训出来是不同的权重。2026-08-26 双集群实测,那个 realization 掉了
8 到 14 个点的闭环 success rate,已回退。详见
docs/DETERMINISTIC_TRAINING.md 的 Realization 事故节。
"""

from __future__ import annotations

import os


def _patch_triton_autotune_pin() -> bool:
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return False

    def _const_bench(self, *args, config, **meta):  # noqa: ANN001
        # 所有 config "耗时"相等 → builtins.min 稳定取 pruned 列表第一个,
        # 选型与 timing 完全解耦;shared-mem pruning 仍生效。benchmark
        # 不再真正执行 kernel,只影响选型不影响正式 launch。
        return (1.0, 1.0, 1.0)

    Autotuner._bench = _const_bench
    return True


def _patch_causal_conv1d_torch_fallback() -> bool:
    """把 causal_conv1d 换成 torch F.conv1d 等价实现(wrapper 同款数值路径)。

    与历史 convfix_wrapper.py 逐行同款:depthwise F.conv1d + silu,
    backward 走 cudnn,在 use_deterministic_algorithms 下确定。seq_idx
    (packed sequence 边界重置)torch 路径不支持,遇到非 None 直接断言
    炸掉,宁可 crash 也不静默改语义。

    为什么不用更快的 .contiguous() 保留 fused kernel:那是另一个
    realization(hash 家族不同),2026-08-26 双集群全量对照掉 8-14 个
    SR 点(contract failure ×3),已回退。本函数的数值行为与 wrapper
    时代逐位等价(flagdet 实证 + 2026-08-27 parity probe),86%/92%
    血统直接延续。
    """
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as _m
    except ImportError:
        return False
    import torch.nn.functional as F

    def _torch_causal_conv1d_fn(
        x=None, weight=None, bias=None, activation=None, seq_idx=None, **kwargs
    ):
        assert seq_idx is None, (
            "deterministic conv1d fallback does not support seq_idx (packed) path"
        )
        assert activation in (None, "silu", "swish"), (
            f"unexpected activation {activation!r}"
        )
        seqlen = x.shape[-1]
        out = F.conv1d(
            x, weight.unsqueeze(1), bias,
            padding=weight.shape[-1] - 1, groups=x.shape[1],
        )[..., :seqlen]
        return F.silu(out) if activation else out

    _m.causal_conv1d_fn = _torch_causal_conv1d_fn
    return True


def enable_deterministic_training(fla_config_dir: str | None = None) -> list[str]:
    """打开全部确定性开关;必须在模型构建与首次 CUDA 计算之前调用。

    返回实际生效的开关名列表(供日志)。fla_config_dir 非空时启用
    FLA 固化选型(FLA_CACHE_MODE=default),既确定又不付保守选型的
    perf 代价;为空则退回 autotune pin(慢但零依赖)。

    注意:fla_config_dir 换的不只是速度,是 kernel 选型 = 换 realization,
    权重 hash 家族随之变。现役 86%/92% 血统用的是 pin(config[0]);正式
    训练不要传这个参数,除非明确要开新 lineage 并重跑闭环 eval 验收。
    """
    applied: list[str] = []

    os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
    applied.append("fa2_deterministic_backward")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    applied.append("cublas_workspace")

    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    applied.append("torch_deterministic_algorithms")

    if fla_config_dir:
        # 固化选型:fla 的 cache 层读取 dump 出的 per-kernel config,
        # 完全跳过现场 benchmark。必须在 fla 首次 import 前设好
        # (fla/ops/utils/cache.py 在模块加载时读 FLA_CACHE_MODE)。
        os.environ["FLA_CONFIG_DIR"] = fla_config_dir
        os.environ.setdefault("FLA_CACHE_MODE", "default")
        applied.append(f"fla_config_dir({fla_config_dir})")
    if _patch_triton_autotune_pin():
        # config 文件缺失或部分 kernel 无 default_config 时的兜底,
        # 保证任何路径都不落回 timing 依赖的选型。
        applied.append("triton_autotune_pinned")

    if _patch_causal_conv1d_torch_fallback():
        applied.append("causal_conv1d_torch_fallback")

    return applied


def enable_inference_numerics_pin(
    fla_config_dir: str | None = None, *, torch_deterministic: bool = False
) -> list[str]:
    """钉死推理 forward 的 kernel 选型;必须在首次 CUDA 计算之前调用。

    解决的是 eval 侧的 decode numerics 分派:同权重同输入的 greedy decode
    在不同机器上会在近平局 token 处 argmax 各走一边,经 history 回灌级联
    成整条轨迹分岔(证据与判决实验见
    docs/EVAL_VERSIONS.md 的 decode numerics 节)。

    默认只钉 FLA/triton 选型:实测这一项让不同机器的 decode 逐字节一致,
    而 decode 吞吐只掉 ~2%。`torch_deterministic=True` 额外打开
    use_deterministic_algorithms 与 CUBLAS_WORKSPACE_CONFIG,那两项在
    decode(memory-bound 的 GEMV)上要付 ~28%,且对跨机一致没有额外贡献,
    所以推理默认不开。训练侧另外两源(FLASH_ATTENTION_DETERMINISTIC、
    causal_conv1d 连续化)只管 backward 的 atomic add,推理用不上。

    **调用方还必须管住 triton 磁盘 cache**:triton 把 autotune 的选型结果
    写进 cache,cache 热时选型直接从盘上读、本函数的 pin 不会被调用。要跨
    机一致就得让每次 eval 用同一份受控 cache(launcher 通过
    TRITON_CACHE_DIR 指定 batch 私有目录)。
    """
    applied: list[str] = []

    if torch_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        applied.append("cublas_workspace")

        import torch

        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        applied.append("torch_deterministic_algorithms")

    if fla_config_dir:
        os.environ["FLA_CONFIG_DIR"] = fla_config_dir
        os.environ.setdefault("FLA_CACHE_MODE", "default")
        applied.append(f"fla_config_dir({fla_config_dir})")
    if _patch_triton_autotune_pin():
        applied.append("triton_autotune_pinned")

    return applied
