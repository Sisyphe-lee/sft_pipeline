"""reasoning / tool_call 两段各自的 token accuracy 与 loss,搭在 TRL 的 chunked loss 上分账;
开了 packing 之后再多一层:按 decision 分账,并把 loss 的归一口径掰回"每条 decision 等权"。

W&B 里多四条曲线:reasoning_token_accuracy、tool_call_token_accuracy、reasoning_loss、
tool_call_loss。哪些 token 算哪一段由 training/vlm_collator.py 的 completion_spans 定
(</think> 及之前 = reasoning,之后 = tool_call)。

为什么不自己再过一遍 lm_head:TRL 1.9.1 的 chunked_nll 是把 lm_head 投影按 256 个有效
token 一块一块做的,永远不落全量 logits,模型 forward 返回的 logits=None。要分段就得拿到
hidden states 再自己算一遍,而 ZeRO-3 下 lm_head 的权重是分片的,再算一遍就要再 all-gather
一遍,每个 rank 的块数还不一样 —— 那条路的坑不值得踩。

这里的做法是把 TRL 那个按块算 loss 的 `_chunk` 换成逐行相同、只多了记账的版本:
`_chunked_cross_entropy_loss` 是按名字从模块全局取 `_chunk` 的,换掉就生效;它先把有效
token 按原顺序打包到前面再切块(stable argsort),所以第 k 块的第 i 个 token 就是原序里第
k*256+i 个有效 token,我们只要按同一个顺序备好每个有效 token 属于哪一段,切片对上就行。
loss 本身的数学一个字不动(同样的 op、同样的顺序),只是顺手在 no_grad 里多算几个 sum。
gradient checkpointing 在 backward 时会把 `_chunk` 重跑一遍,那时 `collecting` 已经关了,
不会重复记账。

**packing(一个 micro-batch 里有 k 条 decision)时多做两件事**,两件都只在 collector 带了
`seg`(每个 token 属于第几条 decision)时才发生:
  1. 记账从 [3] 变成 [3, k],`per_segment()` 给出每条 decision 的九个数,TaskMeter 据此把
     每条归到它自己的 task 上。
  2. loss 按 decision 归一:每个 token 的 nll 先乘 1/(它那条 decision 的有效 token 数),整个
     micro-batch 求和,最后除以 `num_items_in_batch`(训练侧传的是这一步全局的 decision 总数 K)。
     这样每条 decision 的权重都是 1/K,和不 packing 时 turn_mean 的口径逐字对应。
     权重不能走模块全局变量:TRL 用非 reentrant 的 checkpoint,backward 时会把这一块重跑一遍,
     那时账簿已经关了 —— 所以权重是按块切好、当成 checkpoint 的显式实参传进去的。

TRL 不是 1.9.1 那套(找不到 `_chunk` / `_chunked_cross_entropy_loss` 或签名不对)就不打补丁,
训练照跑,只是没有这四条曲线(packing 则会硬报错,它的正确性依赖这个补丁),启动时打一行说明。
"""

from __future__ import annotations

import contextlib
import inspect

import torch
import torch.nn.functional as F

try:
    import trl.trainer.sft_trainer as _sft
except ImportError:  # pragma: no cover - 训练环境一定有 trl
    _sft = None

_ORIGINAL_CHUNK = getattr(_sft, "_chunk", None) if _sft is not None else None
_ORIGINAL_CCE = getattr(_sft, "_chunked_cross_entropy_loss", None) if _sft is not None else None
_CHUNK_PARAMS = ("h", "w", "b", "lbl", "logit_scale", "final_logit_softcapping")
_CCE_PARAMS = ("hidden_states", "lm_head_weight", "chunk_size", "labels", "shift_labels",
               "num_items_in_batch", "logit_scale", "final_logit_softcapping", "lm_head_bias")
_STATE = None      # 当前 compute_loss 的账簿;None 就是不记账,补丁退化成原函数
_PATCHED = False

SPANS = {1: "reasoning", 2: "tool_call"}


def _chunk_with_spans(h, w, b, lbl, logit_scale, final_logit_softcapping):
    # ---- 到 chunk_entropy 为止和 trl 1.9.1 的 _chunk 逐行相同,别改 ----
    with _sft.maybe_gather_lm_head_ctx(w, b):
        logits = h.float() @ w.float().t()
        if b is not None:
            logits = logits + b.float()
    if logit_scale != 1.0:
        logits = logits * logit_scale
    if final_logit_softcapping is not None:
        logits = final_logit_softcapping * torch.tanh(logits / final_logit_softcapping)
    log_p = F.log_softmax(logits, dim=-1)
    chunk_loss = F.nll_loss(log_p, lbl, ignore_index=-100, reduction="sum")
    valid = lbl != -100
    chunk_correct = ((logits.argmax(dim=-1) == lbl) & valid).sum().float()
    chunk_entropy = (-(log_p.exp() * log_p).sum(dim=-1) * valid).sum()
    # ---- 多出来的只有下面这段记账 ----
    state = _STATE
    if state is not None and state.collecting:
        with torch.no_grad():
            n = lbl.shape[0]
            seq = state.seq[state.offset:state.offset + n]
            state.offset += n
            if seq.numel() < n:      # 最后一块的尾巴是打包过去的 -100 位,span 记 0
                seq = torch.cat([seq, seq.new_zeros(n - seq.numel())])
            correct = (logits.argmax(dim=-1) == lbl) & valid
            nll = -log_p.gather(1, lbl.clamp_min(0).unsqueeze(1)).squeeze(1)
            for tag in SPANS:
                mask = valid & (seq == tag)
                state.count[tag] += mask.sum()
                state.correct[tag] += (correct & mask).sum()
                state.loss_sum[tag] += (nll * mask).sum()
    return chunk_loss, chunk_correct, chunk_entropy


def _chunk_packed(h, w, b, lbl, wt, seq, seg, logit_scale, final_logit_softcapping):
    """packing 用的块函数:loss 数学和 `_chunk` 一样,只是每个 token 多乘一个权重(wt 为 None
    时逐字退化成 reduction='sum'),记账按 (span, decision) 两维走显式传进来的 seq / seg。"""
    with _sft.maybe_gather_lm_head_ctx(w, b):
        logits = h.float() @ w.float().t()
        if b is not None:
            logits = logits + b.float()
    if logit_scale != 1.0:
        logits = logits * logit_scale
    if final_logit_softcapping is not None:
        logits = final_logit_softcapping * torch.tanh(logits / final_logit_softcapping)
    log_p = F.log_softmax(logits, dim=-1)
    valid = lbl != -100
    if wt is None:
        chunk_loss = F.nll_loss(log_p, lbl, ignore_index=-100, reduction="sum")
    else:
        # reduction='none' 在 ignore_index 的位置本来就给 0,再乘 wt(那些位也是 0)
        chunk_loss = (F.nll_loss(log_p, lbl, ignore_index=-100, reduction="none") * wt).sum()
    chunk_correct = ((logits.argmax(dim=-1) == lbl) & valid).sum().float()
    chunk_entropy = (-(log_p.exp() * log_p).sum(dim=-1) * valid).sum()
    state = _STATE
    if state is not None and state.collecting:
        with torch.no_grad():
            correct = ((logits.argmax(dim=-1) == lbl) & valid).float()
            nll = -log_p.gather(1, lbl.clamp_min(0).unsqueeze(1)).squeeze(1)
            index = (seq.long() * state.n_seg + seg)[valid]
            state.count_flat.index_add_(0, index, torch.ones_like(index, dtype=state.count_flat.dtype))
            state.correct_flat.index_add_(0, index, correct[valid].to(state.correct_flat.dtype))
            state.loss_flat.index_add_(0, index, nll[valid].to(state.loss_flat.dtype))
    return chunk_loss, chunk_correct, chunk_entropy


def _chunked_cross_entropy_loss_with_state(
    hidden_states,
    lm_head_weight,
    chunk_size,
    labels=None,
    shift_labels=None,
    num_items_in_batch=None,
    logit_scale=1.0,
    final_logit_softcapping=None,
    lm_head_bias=None,
):
    """TRL 1.9.1 `_chunked_cross_entropy_loss` 的逐行拷贝,只在 packing(账簿里有 seg)时改动:
    块函数换成 `_chunk_packed`,并把 per-token 权重 / span / decision 按同一个 argsort 顺序切好传进去。
    没在 packing 就直接调原函数 —— 那条路一个字节都不碰。"""
    state = _STATE
    if state is None or state.seg is None:
        return _ORIGINAL_CCE(hidden_states, lm_head_weight, chunk_size, labels=labels,
                             shift_labels=shift_labels, num_items_in_batch=num_items_in_batch,
                             logit_scale=logit_scale, final_logit_softcapping=final_logit_softcapping,
                             lm_head_bias=lm_head_bias)

    if labels is None and shift_labels is None:
        raise ValueError("At least one of `labels` or `shift_labels` must be provided.")
    if shift_labels is not None:
        hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        labels = shift_labels.reshape(-1)
    else:
        hidden = hidden_states[..., :-1, :].reshape(-1, hidden_states.size(-1))
        labels = labels[..., 1:].reshape(-1)

    valid = labels != -100
    n_valid_tensor = valid.sum()

    correct = hidden.new_zeros((), dtype=torch.float32)
    entropy_sum = hidden.new_zeros((), dtype=torch.float32)
    if n_valid_tensor == 0:
        with _sft.maybe_gather_lm_head_ctx(lm_head_weight, lm_head_bias):
            loss = (hidden_states.float().sum() + lm_head_weight.float().sum()) * 0.0
            if lm_head_bias is not None:
                loss = loss + lm_head_bias.float().sum() * 0.0
        return loss, correct, entropy_sum, n_valid_tensor

    order = valid.to(torch.int8).argsort(descending=True, stable=True)
    hidden = hidden[order]
    labels = labels[order]
    # 账簿里的 full_* 是"按原始位置"的,跟着同一个 order 重排就和 hidden / labels 对齐了
    seq_ordered = state.full_seq.to(hidden.device)[order]
    seg_ordered = state.full_seg.to(hidden.device)[order]
    weight_ordered = None if state.full_weight is None else state.full_weight.to(hidden.device)[order]

    n_padded = (n_valid_tensor / chunk_size).ceil().to(torch.int64) * chunk_size
    loss = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, n_padded, chunk_size):
        h_chunk = hidden[start:start + chunk_size]
        lbl_chunk = labels[start:start + chunk_size]
        chunk_loss, chunk_correct, chunk_entropy = torch.utils.checkpoint.checkpoint(
            _chunk_packed,
            h_chunk,
            lm_head_weight,
            lm_head_bias,
            lbl_chunk,
            None if weight_ordered is None else weight_ordered[start:start + chunk_size],
            seq_ordered[start:start + chunk_size],
            seg_ordered[start:start + chunk_size],
            logit_scale,
            final_logit_softcapping,
            use_reentrant=False,
        )
        loss = loss + chunk_loss
        correct = correct + chunk_correct
        entropy_sum = entropy_sum + chunk_entropy

    if num_items_in_batch is None:
        # 加了权的 loss 已经是"每条 decision 的平均 token loss"之和,除的就该是本地 decision 数
        loss = loss / (state.n_local_decisions if weight_ordered is not None else n_valid_tensor)
    else:
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss, correct, entropy_sum, n_valid_tensor


def install(*, require=False):
    """把补丁装上。装过、或者 TRL 不是预期形态就什么都不做。返回是否已装;require=True 时装不上直接报错。"""
    global _PATCHED
    if _PATCHED:
        return True
    why = None
    if _ORIGINAL_CHUNK is None or _ORIGINAL_CCE is None:
        why = "trl.trainer.sft_trainer 里没有 _chunk / _chunked_cross_entropy_loss"
    elif tuple(inspect.signature(_ORIGINAL_CHUNK).parameters) != _CHUNK_PARAMS:
        why = f"_chunk 签名变了 {tuple(inspect.signature(_ORIGINAL_CHUNK).parameters)}"
    elif tuple(inspect.signature(_ORIGINAL_CCE).parameters) != _CCE_PARAMS:
        why = f"_chunked_cross_entropy_loss 签名变了 {tuple(inspect.signature(_ORIGINAL_CCE).parameters)}"
    if why is not None:
        if require:
            raise RuntimeError(f"packing 要靠 span_metrics 的补丁才能按 decision 归一 loss,但 {why}")
        print(f"[span_metrics] {why},不分段记 accuracy", flush=True)
        return False
    _sft._chunk = _chunk_with_spans
    _sft._chunked_cross_entropy_loss = _chunked_cross_entropy_loss_with_state
    _PATCHED = True
    return True


# ---- 一个 optimizer step 的分段统计:口径跟着 loss_normalization 走 ----
# 每个 span(全部 / reasoning / tool_call)记六列。turn_mean 用"每条 decision 一票"那三列,
# token_mean 用"所有 token 加权"那三列 —— 和训练目标用的是同一种平均,曲线才和 train/loss 对得上。
SPAN_ORDER = ("all", "reasoning", "tool_call")
SPAN_COLS = ("n_tokens", "n_correct", "sum_nll", "sum_decision_accuracy", "sum_decision_loss", "n_decisions")
N_COLS = len(SPAN_ORDER) * len(SPAN_COLS)
METRIC_NAMES = {"all": ("token_accuracy", "loss"),
                "reasoning": ("reasoning_token_accuracy", "reasoning_loss"),
                "tool_call": ("tool_call_token_accuracy", "tool_call_loss")}


def decision_columns(n_reasoning, correct_reasoning, loss_reasoning,
                     n_tool_call, correct_tool_call, loss_tool_call):
    """一条 decision 的十八列。loss / accuracy 的"每条一票"那两列记的是这条自己的均值,配一个条数。"""
    spans = (
        (n_reasoning + n_tool_call, correct_reasoning + correct_tool_call, loss_reasoning + loss_tool_call),
        (n_reasoning, correct_reasoning, loss_reasoning),
        (n_tool_call, correct_tool_call, loss_tool_call),
    )
    out = []
    for n, correct, nll in spans:
        out += [n, correct, nll,
                correct / n if n else 0.0, nll / n if n else 0.0, 1.0 if n else 0.0]
    return out


def decision_rows(spans):
    """把一次 compute_loss 的账簿拆成"每条 decision 十八列"。packing 时 k 条,不 packing 时 1 条。"""
    per_segment = spans.per_segment() if hasattr(spans, "per_segment") else None
    if per_segment is not None:
        # per_segment 的九列:n_all, correct_all, loss_all, n_r, correct_r, loss_r, n_t, correct_t, loss_t
        return [decision_columns(*row[3:9]) for row in per_segment.tolist()]
    return [decision_columns(float(spans.count[1]), float(spans.correct[1]), float(spans.loss_sum[1]),
                             float(spans.count[2]), float(spans.correct[2]), float(spans.loss_sum[2]))]


def metrics_from(total, normalization):
    """十八列 → 六个指标。normalization 就是 recipe 里的 loss_normalization。"""
    if normalization not in {"turn_mean", "token_mean"}:
        raise ValueError(f"unknown loss_normalization: {normalization}")
    out = {}
    for index, span in enumerate(SPAN_ORDER):
        base = index * len(SPAN_COLS)
        accuracy_name, loss_name = METRIC_NAMES[span]
        if normalization == "turn_mean":            # 每条 decision 一票
            denominator = float(total[base + 5])
            numerators = (float(total[base + 3]), float(total[base + 4]))
        else:                                       # 所有 token 加权
            denominator = float(total[base + 0])
            numerators = (float(total[base + 1]), float(total[base + 2]))
        if denominator > 0:
            out[accuracy_name] = numerators[0] / denominator
            out[loss_name] = numerators[1] / denominator
    return out


class StepSpanMeter:
    """整个 optimizer step 的分段账(8 张卡 × accum 个 micro-step 合起来算一次)。

    为什么不像以前那样每个 micro-step 算个比值塞进 Trainer 的 _metrics:那样出来的是"比值的平均",
    accum > 1 时和"平均的比值"对不上;而且跨 rank 求和天然是 token 加权,和 turn_mean 的 loss 口径不一致。
    """

    def __init__(self, normalization="turn_mean"):
        self.normalization = normalization
        self.local = None

    def start_step(self, device):
        self.local = torch.zeros(N_COLS, dtype=torch.float64, device=device)

    def observe(self, spans):
        if self.local is None or spans is None:
            return
        for row in decision_rows(spans):
            self.local += torch.tensor(row, dtype=torch.float64, device=self.local.device)

    def finish_step(self, gather):
        if self.local is None:
            return {}
        total = gather(self.local).reshape(-1, N_COLS).sum(0).cpu()
        self.local = None
        return metrics_from(total, self.normalization)


# train/ 那张表里曲线的名字:总 accuracy 沿用 TRL 那条的名字 mean_token_accuracy(值换成我们算的),
# loss 不塞 —— train/loss 是 HF Trainer 自己那条,同口径同一个数,不去覆盖它。
TRAIN_KEYS = {"token_accuracy": "mean_token_accuracy", "loss": None}


class SpanMetricsCallback:
    """把整步的分段统计塞回 Trainer 的 _metrics —— TRL 的 log() 会把它们连同 loss 一起出到
    master.log 那行和 W&B 的 train/ 表里。每步只塞一个数,所以 TRL 那步"对 list 取平均"是恒等的。"""

    def __init__(self, meter: StepSpanMeter):
        self.meter = meter
        self.trainer = None

    def on_step_begin(self, args, state, control, **kwargs):
        device = self.trainer.accelerator.device if self.trainer is not None else "cpu"
        self.meter.start_step(device)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        for key, value in self.meter.finish_step(self.trainer.accelerator.gather).items():
            name = TRAIN_KEYS.get(key, key)
            if name is None:
                continue
            self.trainer._metrics["train"][name] = [value]   # 直接替换,不 append:TRL 每个
            #                                                  micro-batch 也往同名 key 里塞过东西
        return control

    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda *a, **k: k.get("control")
        raise AttributeError(name)


class SpanMetrics:
    """一次 compute_loss 的账簿。用 SpanMetrics.collect(span, labels[, seg, decision_weights]) 包住
    super().compute_loss。给了 seg 就按 decision 再分一层账(packing)。"""

    def __init__(self, span, labels, seg=None, decision_weights=False):
        shift_span = span[..., 1:].reshape(-1)
        shift_labels = labels[..., 1:].reshape(-1)
        valid = shift_labels != -100
        # TRL 打包后的顺序 = 原序里 valid 的那些依次排在前面;bool 索引正好保序。
        self.seq = shift_span[valid].to(torch.int8)
        self.offset = 0
        self.collecting = True
        size = max(SPANS) + 1
        self._count = torch.zeros(size, device=labels.device)
        self._correct = torch.zeros(size, device=labels.device)
        self._loss_sum = torch.zeros(size, device=labels.device)
        self.seg = None
        self.n_seg = 1
        self.count_flat = self.correct_flat = self.loss_flat = None
        self.full_seq = self.full_seg = self.full_weight = None
        self.n_local_decisions = 1
        if seg is None:
            return
        # ---- packing:按 decision 再分一层 ----
        shift_seg = seg[..., 1:].reshape(-1).to(torch.int64)
        self.n_seg = int(shift_seg.max().item()) + 1 if shift_seg.numel() else 1
        self.seg = shift_seg[valid]
        per_seg = torch.bincount(self.seg, minlength=self.n_seg).to(torch.float32)
        self.n_local_decisions = int((per_seg > 0).sum().item()) or 1
        # 按"原始位置"备好三份全长张量,loss 里跟着 argsort 的 order 一起重排就对齐了
        zeros_i8 = torch.zeros_like(shift_span, dtype=torch.int8)
        self.full_seq = torch.where(valid, shift_span.to(torch.int8), zeros_i8)
        self.full_seg = torch.where(valid, shift_seg, torch.zeros_like(shift_seg))
        if decision_weights:
            weight = torch.where(per_seg > 0, 1.0 / per_seg.clamp(min=1.0), torch.zeros_like(per_seg))
            self.full_weight = torch.where(valid, weight.to(labels.device)[shift_seg],
                                           torch.zeros(shift_seg.shape, device=labels.device))
        flat = size * self.n_seg
        self.count_flat = torch.zeros(flat, device=labels.device, dtype=torch.float32)
        self.correct_flat = torch.zeros(flat, device=labels.device, dtype=torch.float32)
        self.loss_flat = torch.zeros(flat, device=labels.device, dtype=torch.float32)

    # packing 时这三个从 [3, k] 的账上汇总;不 packing 就是 _chunk_with_spans 直接写的那三个。
    @property
    def count(self):
        return self._count if self.count_flat is None else self.count_flat.view(-1, self.n_seg).sum(1)

    @property
    def correct(self):
        return self._correct if self.correct_flat is None else self.correct_flat.view(-1, self.n_seg).sum(1)

    @property
    def loss_sum(self):
        return self._loss_sum if self.loss_flat is None else self.loss_flat.view(-1, self.n_seg).sum(1)

    @classmethod
    @contextlib.contextmanager
    def collect(cls, span, labels, seg=None, decision_weights=False):
        global _STATE
        if span is None or labels is None or not install(require=seg is not None):
            yield None
            return
        state = cls(span, labels, seg=seg, decision_weights=decision_weights)
        _STATE = state
        try:
            yield state
        finally:
            state.collecting = False
            _STATE = None

    def reduce(self, gather):
        """跨 rank 求和后给出四个数。gather 是 accelerator.gather(不是 gather_for_metrics,那个在 epoch 末会截断)。"""
        size = self._count.numel()
        count = gather(self.count).reshape(-1, size).sum(0)
        correct = gather(self.correct).reshape(-1, size).sum(0)
        loss_sum = gather(self.loss_sum).reshape(-1, size).sum(0)
        out = {}
        for tag, name in SPANS.items():
            if count[tag] > 0:
                out[f"{name}_token_accuracy"] = (correct[tag] / count[tag]).item()
                out[f"{name}_loss"] = (loss_sum[tag] / count[tag]).item()
        return out

    def per_segment(self):
        """packing 时给每条 decision 的九个数,列的顺序和 training/task_metrics.py 的 COLS 一致。
        不 packing 返回 None。**不跨 rank**:各 rank 的 decision 条数不一样,gather 不了 —— 由
        TaskMeter 先按 task 汇总成同形状的表再 gather。"""
        if self.count_flat is None:
            return None
        count = self.count_flat.view(-1, self.n_seg)
        correct = self.correct_flat.view(-1, self.n_seg)
        loss = self.loss_flat.view(-1, self.n_seg)
        return torch.stack([
            count[1] + count[2], correct[1] + correct[2], loss[1] + loss[2],
            count[1], correct[1], loss[1],
            count[2], correct[2], loss[2],
        ], dim=1).to(torch.float64)
