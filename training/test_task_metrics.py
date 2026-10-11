"""task/ 表:六个指标的平均方式跟着 loss_normalization 走(turn_mean = 每条 decision 一票,
和 train/loss 同口径;token_mean = 所有 token 加权),每步只出这一步的原始数,commit=False。"""

from types import SimpleNamespace

import torch

from training import task_metrics as T


def _spans(n1, c1, l1, n2, c2, l2):
    z = lambda *v: torch.tensor([0.0, *v], dtype=torch.float64)   # index 0 空着,1 = reasoning,2 = tool_call
    return SimpleNamespace(count=z(n1, n2), correct=z(c1, c2), loss_sum=z(l1, l2))


def test_turn_mean_gives_every_decision_one_vote():
    meter = T.TaskMeter(["a", "b"], ["a", "b", "a"], normalization="turn_mean")
    identity = lambda t: t
    meter.start_step("cpu")
    # row 0(task a):reasoning 10 个 token 对 5 个、nll 和 20;tool_call 4 个对 4 个、nll 和 0.4
    meter.observe(_spans(10, 5, 20.0, 4, 4, 0.4), torch.tensor([0]))
    # row 1(task b)
    meter.observe(_spans(2, 1, 2.0, 2, 1, 1.0), torch.tensor([1]))
    out = meter.finish_step(identity)
    assert out["task/a/reasoning_token_accuracy"] == 0.5          # 5/10
    assert out["task/a/tool_call_token_accuracy"] == 1.0
    assert out["task/a/token_accuracy"] == 9 / 14                 # 只有一条 decision:就是它自己的 accuracy
    assert abs(out["task/a/loss"] - 20.4 / 14) < 1e-12            # 只有一条 decision:就是它自己的均值
    assert abs(out["task/a/reasoning_loss"] - 2.0) < 1e-12
    assert abs(out["task/a/tool_call_loss"] - 0.1) < 1e-12
    assert out["task/b/token_accuracy"] == 0.5

    # 同一个 task 两条 decision:loss 与 accuracy 都是两条各自均值的**等权**平均
    meter.start_step("cpu")
    meter.observe(_spans(10, 5, 30.0, 0, 0, 0.0), torch.tensor([0]))     # loss 3.0、accuracy 0.5,10 个 token
    meter.observe(_spans(1, 1, 1.0, 0, 0, 0.0), torch.tensor([2]))       # loss 1.0、accuracy 1.0,1 个 token
    out = meter.finish_step(identity)
    assert abs(out["task/a/loss"] - 2.0) < 1e-12                          # (3.0 + 1.0) / 2
    assert abs(out["task/a/reasoning_loss"] - 2.0) < 1e-12
    assert abs(out["task/a/token_accuracy"] - 0.75) < 1e-12               # (0.5 + 1.0) / 2
    assert "task/b/token_accuracy" not in out                             # 这一步没出现的 task 不更新


def test_token_mean_weights_every_token():
    """换成 token_mean:同一批账,六个指标全变成 token 加权 —— loss 和统计跟着同一个开关走。"""
    meter = T.TaskMeter(["a"], ["a", "a"], normalization="token_mean")
    meter.start_step("cpu")
    meter.observe(_spans(10, 5, 30.0, 0, 0, 0.0), torch.tensor([0]))
    meter.observe(_spans(1, 1, 1.0, 0, 0, 0.0), torch.tensor([1]))
    out = meter.finish_step(lambda t: t)
    assert abs(out["task/a/loss"] - 31.0 / 11) < 1e-12
    assert abs(out["task/a/token_accuracy"] - 6 / 11) < 1e-12


def test_no_sliding_window_each_step_is_raw():
    meter = T.TaskMeter(["a"], ["a"])
    identity = lambda t: t
    meter.start_step("cpu")
    meter.observe(_spans(10, 10, 0.0, 0, 0, 0.0), torch.tensor([0]))
    assert meter.finish_step(identity)["task/a/reasoning_token_accuracy"] == 1.0
    meter.start_step("cpu")
    meter.observe(_spans(10, 0, 50.0, 0, 0, 0.0), torch.tensor([0]))
    out = meter.finish_step(identity)
    assert out["task/a/reasoning_token_accuracy"] == 0.0     # 上一步的 10/10 不掺进来
    assert out["task/a/reasoning_loss"] == 5.0
    meter.start_step("cpu")
    assert meter.finish_step(identity) == {}                 # 没观测就没输出


def test_matches_train_loss_for_a_single_task_step():
    """单 task 时 task/<task>/loss 就该等于 train/loss —— train/loss 是这一步全部 decision 的等权平均。"""
    meter = T.TaskMeter(["a"], ["a"] * 4)
    per_decision = [(120, 60, 180.0), (30, 20, 15.0), (400, 390, 40.0), (7, 3, 14.0)]
    meter.start_step("cpu")
    for row, (n, correct, loss_sum) in enumerate(per_decision):
        meter.observe(_spans(n, correct, loss_sum, 0, 0, 0.0), torch.tensor([row]))
    out = meter.finish_step(lambda t: t)
    train_loss = sum(loss_sum / n for n, _, loss_sum in per_decision) / len(per_decision)
    assert abs(out["task/a/loss"] - train_loss) < 1e-12
    token_weighted = sum(loss_sum for _, _, loss_sum in per_decision) / sum(n for n, _, _ in per_decision)
    assert abs(token_weighted - train_loss) > 0.1            # 和旧口径确实不是一个数


def test_callback_logs_own_section_commit_false(monkeypatch):
    meter = T.TaskMeter(["a"], ["a"])
    cb = T.TaskMetricsCallback(meter)
    logged = []
    monkeypatch.setattr(cb, "_wandb_run", lambda: SimpleNamespace(log=lambda row, commit=True: logged.append((row, commit))))
    cb.trainer = SimpleNamespace(accelerator=SimpleNamespace(device="cpu", gather=lambda t: t),
                                 is_world_process_zero=lambda: True)
    cb.on_step_begin(None, None, "ctl")
    meter.observe(_spans(3, 3, 0.0, 1, 0, 2.0), torch.tensor([0]))
    assert cb.on_step_end(None, SimpleNamespace(global_step=5), "ctl") == "ctl"
    row, commit = logged[0]
    assert commit is False and row["train/global_step"] == 5
    assert set(k for k in row if k.startswith("task/")) == {f"task/a/{m}" for m in T.METRICS}


def test_step_span_meter_matches_task_table_for_a_single_task():
    """单 task 的 run 里 train/ 那几条和 task/<task>/ 那几条必须逐位相同 —— 同一套账、同一个口径。"""
    from training.span_metrics import StepSpanMeter
    identity = lambda t: t
    decisions = [(_spans(120, 60, 180.0, 30, 29, 3.0), 0),
                 (_spans(30, 20, 15.0, 8, 8, 0.8), 1),
                 (_spans(400, 390, 40.0, 60, 55, 12.0), 2)]
    for normalization in ("turn_mean", "token_mean"):
        task = T.TaskMeter(["solo"], ["solo"] * 3, normalization=normalization)
        step = StepSpanMeter(normalization=normalization)
        task.start_step("cpu"); step.start_step("cpu")
        for spans, row in decisions:
            task.observe(spans, torch.tensor([row]))
            step.observe(spans)
        from_task = {k.split("/")[-1]: v for k, v in task.finish_step(identity).items()}
        from_step = step.finish_step(identity)
        assert set(from_task) == set(from_step) == set(T.METRICS)
        for key in from_step:
            assert abs(from_step[key] - from_task[key]) < 1e-12, key
    # 两个口径确实不是一个数
    task_turn = T.TaskMeter(["solo"], ["solo"] * 3, normalization="turn_mean")
    task_token = T.TaskMeter(["solo"], ["solo"] * 3, normalization="token_mean")
    for meter in (task_turn, task_token):
        meter.start_step("cpu")
        for spans, row in decisions:
            meter.observe(spans, torch.tensor([row]))
    turn, token = meter_out(task_turn, identity), meter_out(task_token, identity)
    assert abs(turn["loss"] - token["loss"]) > 0.1
    assert abs(turn["token_accuracy"] - token["token_accuracy"]) > 0.05


def meter_out(meter, gather):
    return {k.split("/")[-1]: v for k, v in meter.finish_step(gather).items()}
