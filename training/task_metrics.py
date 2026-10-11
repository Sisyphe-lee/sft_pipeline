"""W&B 里单开一张 task/ 表:每个 task 六条曲线,每个 optimizer step 更新一次。

六条:token_accuracy、reasoning_token_accuracy、tool_call_token_accuracy、loss、reasoning_loss、
tool_call_loss。reasoning / tool_call 的分界和 training/span_metrics.py 一样(</think> 及之前 =
reasoning),不带前缀的那两条是两段合起来。task 列表从训练 manifest 的 task_id 来,不写死。

口径(2026-09-17 按用户要求统一的):**六个指标的平均方式跟着 recipe 的 loss_normalization 走**,
和 train/ 那张表用同一套公式(training/span_metrics.py 的 metrics_from):
  * turn_mean(现役配方):每条 decision 一票 —— 每条先算自己的平均 token loss / 自己的 accuracy,
    再在这个 task 的 decision 之间等权平均。训练目标就是这么算的,所以单 task 的 run 里
    task/<task>/* 和 train/* 六条应当逐位相等。
  * token_mean:所有 token 加权(Σnll / Σtoken、Σcorrect / Σcount)。
以后要 ablate loss 的归一方式,改一个 loss_normalization,loss 和统计一起翻过去。
**不做滑动平均**:每一步就是这一步的原始数(用户要原始数据,要平滑在 W&B 界面上开)。

账怎么记:span_metrics 那本账(SpanMetrics)已经按 reasoning / tool_call 记了本 micro-batch 的
count / correct / loss_sum,decision_rows() 把它拆成"每条 decision 十八列",这里按 task 累加。
不开 packing 时一个 micro-batch 就是一条 decision;开了 packing 一个 micro-batch 是一个 pack,
per_segment() 给出每条 decision 的账,逐条归到各自的 task 上。

跨 rank:每个 micro-batch 只累在本 rank 的 [n_task, 18] 表里,每个 optimizer step 结束时
gather 一次(callback 在所有 rank 上都跑,gather 是对称的),主进程 wandb.log。
没有 W&B 时什么都不记 —— 这张表几十条曲线,塞进 master.log 没意义。
"""

from __future__ import annotations

import torch

from training.span_metrics import METRIC_NAMES, N_COLS, SPAN_COLS, SPAN_ORDER, decision_rows, metrics_from

WANDB_SECTION = "task"
COLS = tuple(f"{span}_{col}" for span in SPAN_ORDER for col in SPAN_COLS)
METRICS = tuple(name for span in SPAN_ORDER for name in METRIC_NAMES[span])


class TaskMeter:
    def __init__(self, tasks, task_of_row, normalization="turn_mean"):
        self.tasks = list(tasks)
        self.index = {t: i for i, t in enumerate(self.tasks)}
        self.task_of_row = list(task_of_row)
        self.normalization = normalization
        self.local = None          # 本 rank 本步的 [n_task, 18]
        self._warned = False

    def start_step(self, device):
        self.local = torch.zeros(len(self.tasks), N_COLS, dtype=torch.float64, device=device)

    def observe(self, spans, row_index):
        """把一个 micro-batch 的 span 账归到它的 task 上。spans 是 SpanMetrics;row_index 是这条 decision
        在 manifest 里的序号(tensor 或 int);packing 时是这一个 pack 里 k 条 decision 的序号。"""
        if self.local is None or spans is None:
            return
        rows = decision_rows(spans)
        if isinstance(row_index, torch.Tensor):
            indices = row_index.reshape(-1).tolist()
        elif hasattr(row_index, "__len__"):
            indices = list(row_index)
        else:
            indices = [row_index]
        if len(indices) != len(rows):
            if len(rows) > 1:
                raise ValueError(f"这个 micro-batch 有 {len(rows)} 条 decision,但给了 {len(indices)} 个 row_index")
            if not self._warned:
                print("[task_metrics] 一个 micro-batch 里有多条 decision 却只有一条账,"
                      "只按第一条归 task,数字是错的", flush=True)
                self._warned = True
            indices = indices[:1]
        for row, index in zip(rows, indices):
            task = self.index.get(self.task_of_row[int(index)])
            if task is not None:
                self.local[task] += torch.tensor(row, dtype=torch.float64, device=self.local.device)

    def finish_step(self, gather):
        """gather 各 rank 的本步表,返回 {task/<task>/<metric>: 值}(只含这一步出现过的 task)。"""
        if self.local is None:
            return {}
        total = gather(self.local).reshape(-1, len(self.tasks), N_COLS).sum(0).cpu()
        self.local = None
        out = {}
        for i in (total[:, 0] > 0).nonzero().flatten().tolist():
            for metric, value in metrics_from(total[i], self.normalization).items():
                out[f"{WANDB_SECTION}/{self.tasks[i]}/{metric}"] = value
        return out


class TaskMetricsCallback:
    """挂在 Trainer 上。要在 Trainer 建好之后把 .trainer 指过来。"""

    def __init__(self, meter: TaskMeter):
        self.meter = meter
        self.trainer = None

    def on_step_begin(self, args, state, control, **kwargs):
        device = self.trainer.accelerator.device if self.trainer is not None else "cpu"
        self.meter.start_step(device)
        return control

    def _wandb_run(self):
        try:
            import wandb
        except ImportError:
            return None
        return wandb.run

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        row = self.meter.finish_step(self.trainer.accelerator.gather)   # 不用 gather_for_metrics:epoch 末会被截断
        run = self._wandb_run()
        if row and run is not None and self.trainer.is_world_process_zero():
            row["train/global_step"] = int(state.global_step)
            run.log(row, commit=False)
        return control

    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda *a, **k: k.get("control")
        raise AttributeError(name)
