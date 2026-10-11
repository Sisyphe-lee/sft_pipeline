"""MFU 账簿:FLOPs 公式、峰值查表(型号 × 精度)、callback 把指标塞进 TRL 的 _metrics。"""

import collections
from types import SimpleNamespace

import torch

from training import mfu as M


def test_sparse_model_reports_measured_throughput_without_dense_flops():
    meter = _meter()
    meter.report_flops = False
    meter.peak_tflops = 100.
    meter.start_step()
    meter.observe({'input_ids': torch.tensor([[1, 2, 3]])})
    metrics = meter.finish_step()
    assert metrics['tokens_per_step'] == 3
    assert metrics['tokens_per_sec'] > 0
    assert 'mfu' not in metrics and 'model_tflops_per_gpu' not in metrics


def test_peak_table_picks_model_and_precision(monkeypatch):
    monkeypatch.delenv("MFU_PEAK_TFLOPS", raising=False)
    # robbys7 的 "L20D" 标签下面是 B300(cc 10.3);真 L20 是 cc 8.9
    assert M.peak_tflops_for("NVIDIA L20D", (10, 3), "bf16")[0] == 2250.0
    assert M.peak_tflops_for("NVIDIA L20D", (8, 9), "bf16")[0] == 119.5
    assert M.peak_tflops_for("NVIDIA L20D", (10, 3), "fp8")[0] == 4500.0
    assert M.peak_tflops_for("NVIDIA L20D", (10, 3), "tf32")[0] == 1125.0
    assert M.peak_tflops_for("NVIDIA H100 80GB HBM3", (9, 0), "bf16")[0] == 989.0
    value, note = M.peak_tflops_for("NVIDIA RTX 6000 Ada", (8, 9), "bf16")
    assert value is None and "不在" in note
    monkeypatch.setenv("MFU_PEAK_TFLOPS", "1000")
    assert M.peak_tflops_for("whatever", (0, 0), "bf16")[0] == 1000.0


def test_detect_precision_prefers_training_args():
    assert M.detect_precision(SimpleNamespace(bf16=True, fp16=False)) == "bf16"
    assert M.detect_precision(SimpleNamespace(bf16=False, fp16=True)) == "fp16"
    assert M.detect_precision(SimpleNamespace(bf16=False, fp16=False), allow_tf32=True) == "tf32"
    assert M.detect_precision(SimpleNamespace(bf16=False, fp16=False), allow_tf32=False) == "fp32"


def _meter(world=1):
    return M.MfuMeter(n_text=1000.0, n_vision=100.0, n_full_attention_layers=2, d_attn=8,
                      image_token_id=7, world_size=world, gather=lambda t: t)


def test_step_flops_formula_and_metrics():
    meter = _meter()
    meter.peak_tflops, meter.precision = 1e-9, "bf16"   # 峰值取极小,让 mfu 数值好核
    meter.start_step()
    ids = torch.tensor([[1, 7, 7, 2, 0], [3, 4, 0, 0, 0]])        # 两条样本,长 4 与 2(按 mask)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    meter.observe({"input_ids": ids, "attention_mask": mask})
    assert meter.tokens == 6 and meter.image_tokens == 2 and meter.seq_sq == 16 + 4
    expected = 6 * 1000 * 6 + 2 * 100 * 2 + 6 * 2 * 8 * 20
    assert meter.step_flops() == expected
    out = meter.finish_step()
    assert out["tokens_per_step"] == 6
    assert abs(out["model_tflops_per_gpu"] * 1e12 * out["step_time_s"] - expected) < 1e-3
    assert abs(out["mfu"] - out["model_tflops_per_gpu"] / meter.peak_tflops) < 1e-9


def _trainer():
    return SimpleNamespace(_metrics={"train": collections.defaultdict(list), "eval": collections.defaultdict(list)},
                           model=SimpleNamespace(training=True), is_world_process_zero=lambda: True)


def test_callback_falls_back_to_trl_metrics_without_wandb(monkeypatch):
    meter = _meter()
    meter.peak_tflops, meter.precision = 100.0, "bf16"
    cb = M.MfuCallback(meter)
    monkeypatch.setattr(cb, "_wandb_run", lambda: None)
    trainer = _trainer()
    cb.trainer = trainer
    cb.on_step_begin(None, None, "ctl")
    meter.observe({"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])})
    assert cb.on_step_end(None, None, "ctl") == "ctl"
    got = trainer._metrics["train"]
    for key in ("mfu", "tokens_per_sec", "tokens_per_step", "step_time_s", "model_tflops_per_gpu"):
        assert len(got[key]) == 1, key
    assert got["tokens_per_step"][0] == 3
    # 没观测到样本的 step 不写任何东西
    cb.on_step_begin(None, None, "ctl")
    cb.on_step_end(None, None, "ctl")
    assert len(got["mfu"]) == 1


def test_callback_logs_its_own_wandb_section(monkeypatch):
    meter = _meter()
    meter.peak_tflops, meter.precision = 100.0, "bf16"
    cb = M.MfuCallback(meter)
    logged = []
    fake_run = SimpleNamespace(log=lambda row, commit=True: logged.append((row, commit)))
    monkeypatch.setattr(cb, "_wandb_run", lambda: fake_run)
    trainer = _trainer()
    cb.trainer = trainer
    cb.on_step_begin(None, None, "ctl")
    meter.observe({"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])})
    cb.on_step_end(None, SimpleNamespace(global_step=7), "ctl")
    assert len(logged) == 1
    row, commit = logged[0]
    assert commit is False                      # 挂到 Trainer 即将提交的那一行上
    assert row["train/global_step"] == 7
    assert {"mfu/mfu", "mfu/tokens_per_sec", "mfu/tokens_per_step", "mfu/step_time_s", "mfu/model_tflops_per_gpu"} <= set(row)
    assert not trainer._metrics["train"]        # 有 W&B 时不再往 train/ 里塞,免得两张表重复
    # 非主进程不发
    trainer.is_world_process_zero = lambda: False
    cb.on_step_begin(None, None, "ctl")
    meter.observe({"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])})
    cb.on_step_end(None, SimpleNamespace(global_step=8), "ctl")
    assert len(logged) == 1
