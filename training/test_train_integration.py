"""CPU integration: real tiny Qwen VLM, sharded weights, Trainer and processor.

This exercises plumbing; it does not qualify full-size GPU/hybrid-attention runs.
"""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from datasets import Dataset
from PIL import Image
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from trl import SFTConfig
import yaml

from data_pipeline.test_sft_selection import row, select
from training.test_history_truncation import make_processor
from training.train_sft import FiniteSFTTrainer
from training.vlm_collator import VlmCollator


def test_zero3_init_is_explicit_and_recipe_selectable(tmp_path):
    from training.train_sft import build_parser
    from training.recipe import parse_recipe

    argv = ['--model', '/model', '--data-manifest', '/manifest', '--output', '/output']
    assert 'zero3_init' not in vars(build_parser().parse_args(['--recipe', '/recipe'] + argv))
    recipe = tmp_path / 'partitioned.yaml'
    recipe.write_text('zero3_init: true\n')
    assert parse_recipe(build_parser(), ['--recipe', str(recipe)] + argv).zero3_init
    assert not parse_recipe(
        build_parser(), ['--recipe', str(recipe)] + argv + ['--no-zero3-init']
    ).zero3_init


def tiny_model(processor):
    tokenizer = processor.tokenizer
    config = Qwen3_5Config(
        text_config=dict(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, layer_types=['full_attention', 'full_attention'],
                         max_position_embeddings=2048,
                         rope_parameters=dict(rope_type='default', rope_theta=10000.,
                                              partial_rotary_factor=1., mrope_section=[1, 1, 2]),
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=32, num_position_embeddings=256, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2),
        **{name + '_token_id': tokenizer.convert_tokens_to_ids(token)
           for name, token in [('image', '<|image_pad|>'), ('video', '<|video_pad|>'),
                               ('vision_start', '<|vision_start|>'), ('vision_end', '<|vision_end|>')]},
    )
    config._attn_implementation = 'eager'
    return Qwen3_5ForConditionalGeneration(config)


def examples(root, count=12):
    Image.new('RGB', (64, 64), 'red').save(root / 'image.png')
    rows = [row(seed=i, episode=f'rollout-{i}') for i in range(count)]
    for i, r in enumerate(rows):
        r['prompt'][0]['content'] = [dict(type='text', text='current'), dict(type='image')]
        r['images'] = ['image.png']
        r['completion'][0]['reasoning_content'] = 'target_reasoning ' * (1 + i % 3 * 15)
    return rows


@pytest.mark.parametrize('validation', [False, True])
def test_clean_manifest_recipe_train_validation_and_resume(tmp_path, monkeypatch, validation):
    import training.train_sft as entry

    torch.set_num_threads(1)
    processor = make_processor()
    model_path = tmp_path / 'model'
    model = tiny_model(processor)
    model.save_pretrained(model_path, max_shard_size='10KB')
    processor.save_pretrained(model_path)
    assert (model_path / 'model.safetensors.index.json').is_file()
    output, _ = select(tmp_path, examples(tmp_path), dict(validation_fraction=.5))
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text(yaml.safe_dump(dict(
        max_steps=2, global_batch_size=2, save_steps=1, save_total_limit=3,
        max_input_tokens=1024, deepspeed=None, attn_implementation='eager',
        gradient_checkpointing=False, validation=validation, eval_steps=1,
        sampling_strategy='episode_uniform' if validation else 'turn_uniform',
    )))
    argv = ['train', '--recipe', str(recipe), '--model', str(model_path),
            '--data-manifest', str(output), '--output', str(tmp_path / 'run')]
    # Override only device/dtype for this CPU test; use real loading and training.
    monkeypatch.setattr(entry, 'SFTConfig', lambda **kw: SFTConfig(**(kw | dict(use_cpu=True, bf16=False))))
    original_loader = entry.AutoModelForImageTextToText.from_pretrained
    monkeypatch.setattr(entry.AutoModelForImageTextToText, 'from_pretrained',
                        lambda *a, **kw: original_loader(*a, **(kw | dict(dtype='float32'))))
    monkeypatch.setattr(sys, 'argv', argv)
    entry.main()
    run = tmp_path / 'run'
    state = json.loads((run / 'trainer_state.json').read_text())
    assert state['global_step'] == 2
    train_logs = [log for log in state['log_history'] if 'loss' in log]
    assert train_logs
    for log in train_logs:
        assert {'reasoning_loss', 'tool_call_loss', 'reasoning_token_accuracy',
                'tool_call_token_accuracy', 'mean_token_accuracy', 'tokens_per_sec'} <= log.keys()
    assert any('eval_loss' in log for log in state['log_history']) == validation
    assert (run / 'eval_results.json').exists() == validation
    record = json.loads((run / 'resolved_config.json').read_text())
    # 归一口径跟随 recipe 的 loss_normalization,默认 turn_mean(老入口与所有现役 run 的实际口径,
    # 见 FiniteSFTTrainer 注释);token_mean 是 opt-in。
    assert record['loss']['normalization'] == 'turn_mean'
    assert 'strategy' in record['sampling']
    assert record['length_policy']['train']['retained'] > 0
    assert json.loads((run / 'length_policy_plan.json').read_text())['version'] == 1
    original = Qwen3_5ForConditionalGeneration.from_pretrained(run).state_dict()
    monkeypatch.setattr(
        entry, 'prepare_length_policy_distributed',
        lambda *args, **kwargs: pytest.fail('resume should reuse the cached length plan'),
    )
    monkeypatch.setattr(sys, 'argv', argv + ['--resume-from-checkpoint', str(run / 'checkpoint-1')])
    entry.main()
    resumed = Qwen3_5ForConditionalGeneration.from_pretrained(run).state_dict()
    assert all(torch.equal(original[key], resumed[key]) for key in original)
    resumed_history = json.loads((run / 'trainer_state.json').read_text())['log_history']
    resumed_logs = [log for log in resumed_history if 'loss' in log]
    assert resumed_logs[-1]['num_tokens'] == train_logs[-1]['num_tokens']


def test_chunked_loss_and_gradients_match_turn_mean_with_unequal_lengths(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    processor = make_processor()
    model = tiny_model(processor)
    reference = copy.deepcopy(model)
    rows = [dict(row_json=json.dumps(r), row_index=i) for i, r in enumerate(examples(tmp_path, 2))]
    collator = VlmCollator(processor, tmp_path, 1024)
    config = SFTConfig(output_dir=str(tmp_path / 'loss'), use_cpu=True, bf16=False,
                       per_device_train_batch_size=1, per_device_eval_batch_size=1,
                       gradient_accumulation_steps=2, gradient_checkpointing=False,
                       max_length=None, remove_unused_columns=False, report_to='none',
                       dataset_kwargs={'skip_prepare_dataset': True}, loss_type='chunked_nll')
    # 这条测的是 turn_mean 口径(每条 decision 取自己的 token 均值),也是默认值;这里显式写出来。
    trainer = FiniteSFTTrainer(model=model, args=config, train_dataset=Dataset.from_list(rows),
                               loss_normalization='turn_mean',
                               processing_class=processor, data_collator=collator)
    batches = [collator([r]) for r in rows]
    counts = [(batch['labels'][:, 1:] != -100).sum().item() for batch in batches]
    assert counts[0] != counts[1]
    for batch in batches:
        actual = trainer.compute_loss(model, dict(batch), num_items_in_batch=torch.tensor(sum(counts))) / 2
        logits = reference(**{k: v for k, v in batch.items() if k not in {'labels', '_sft_row_index', '_sft_span'}}, use_cache=False).logits
        expected = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   batch['labels'][:, 1:].reshape(-1), ignore_index=-100) / 2
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        actual.backward()
        expected.backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-5, rtol=1e-5)


def _zero_supervision_rank(rank, rendezvous):
    from datetime import timedelta

    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        trainer = SimpleNamespace(accelerator=SimpleNamespace(device=torch.device('cpu')))
        labels = torch.tensor([[-100, -100 if rank == 0 else 1]])
        with pytest.raises(ValueError, match='at least one rank'):
            FiniteSFTTrainer.compute_loss(trainer, None, dict(labels=labels))
    finally:
        dist.destroy_process_group()


def test_zero_supervision_stops_all_ranks_before_model_forward(tmp_path):
    mp.spawn(_zero_supervision_rank, args=(str(tmp_path / 'rendezvous'),), nprocs=2, join=True)


def test_span_metrics_split_supervised_tokens_without_changing_loss(tmp_path):
    """reasoning / tool_call 分账:两段的 token 数与 correct 数加起来就是 TRL 自己数的,
    loss 与不打补丁时逐位相同。"""
    torch.set_num_threads(1)
    torch.manual_seed(23)
    processor = make_processor()
    model = tiny_model(processor)
    reference = copy.deepcopy(model)
    rows = [dict(row_json=json.dumps(r), row_index=i) for i, r in enumerate(examples(tmp_path, 2))]
    collator = VlmCollator(processor, tmp_path, 1024)
    config = SFTConfig(output_dir=str(tmp_path / 'span'), use_cpu=True, bf16=False,
                       per_device_train_batch_size=1, per_device_eval_batch_size=1,
                       gradient_checkpointing=False, max_length=None, remove_unused_columns=False,
                       report_to='none', dataset_kwargs={'skip_prepare_dataset': True},
                       loss_type='chunked_nll')
    trainer = FiniteSFTTrainer(model=model, args=config, train_dataset=Dataset.from_list(rows),
                               processing_class=processor, data_collator=collator)
    think_end = processor.tokenizer.convert_tokens_to_ids('</think>')
    for r in rows:
        batch = collator([r])
        span, labels = batch['_sft_span'], batch['labels']
        supervised = labels != -100
        assert ((span > 0) == supervised).all()
        assert (span == 1).sum() > 0 and (span == 2).sum() > 0
        cut = int(((batch['input_ids'][0] == think_end) & supervised[0]).nonzero()[0])
        assert span[0, cut] == 1 and span[0, cut + 1] == 2 and (span[0, cut + 1:] != 1).all()

        loss, outputs = trainer.compute_loss(model, dict(batch), return_outputs=True)
        spans = trainer._last_spans
        assert spans is not None
        torch.testing.assert_close(spans.count[1] + spans.count[2], outputs.num_valid_tokens.float())
        torch.testing.assert_close(spans.correct[1] + spans.correct[2], outputs.num_correct_tokens)
        metrics = trainer._metrics['train']
        for key in ('reasoning_token_accuracy', 'tool_call_token_accuracy',
                    'reasoning_loss', 'tool_call_loss'):
            assert metrics[key], key
            assert 0.0 <= metrics[key][-1] or key.endswith('_loss')
        # loss 的数学没动:和直接算 cross_entropy 的参考值一致
        logits = reference(**{k: v for k, v in batch.items()
                              if k not in {'labels', '_sft_row_index', '_sft_span'}}, use_cache=False).logits
        expected = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)
        # 两段各自的 loss 按 token 数加权平均回去,就是整段的 loss
        n1, n2 = spans.count[1], spans.count[2]
        recombined = (metrics['reasoning_loss'][-1] * n1 + metrics['tool_call_loss'][-1] * n2) / (n1 + n2)
        torch.testing.assert_close(recombined.float(), expected.float(), atol=1e-4, rtol=1e-4)


def test_restore_token_counter_reads_last_logged_num_tokens(tmp_path):
    from training.train_sft import restore_token_counter
    ckpt = tmp_path / 'checkpoint-68'
    ckpt.mkdir()
    (ckpt / 'trainer_state.json').write_text(json.dumps({'global_step': 68, 'log_history': [
        {'step': 67, 'loss': 1.0, 'num_tokens': 7.696e7},
        {'step': 68, 'loss': 0.9, 'num_tokens': 7.711e7},
        {'step': 68, 'train_runtime': 1.0},          # 汇总条目没有 num_tokens,要跳过
    ]}))
    trainer = SimpleNamespace(_total_train_tokens=0)
    assert restore_token_counter(trainer, ckpt) == 77110000
    assert trainer._total_train_tokens == 77110000
    # 没有 num_tokens 的 log_history:不改、不抛
    (ckpt / 'trainer_state.json').write_text(json.dumps({'global_step': 1, 'log_history': [{'step': 1}]}))
    trainer = SimpleNamespace(_total_train_tokens=0)
    assert restore_token_counter(trainer, ckpt) is None and trainer._total_train_tokens == 0
    assert restore_token_counter(trainer, tmp_path / 'nope') is None


def test_mean_token_accuracy_is_replaced_by_our_own():
    """TRL 那条 mean_token_accuracy 的值被换成我们整步算的(曲线名不变,口径跟着 loss_normalization)。"""
    from collections import defaultdict
    from types import SimpleNamespace

    from training.span_metrics import SpanMetricsCallback, StepSpanMeter

    meter = StepSpanMeter(normalization="turn_mean")
    callback = SpanMetricsCallback(meter)
    trainer = SimpleNamespace(accelerator=SimpleNamespace(device="cpu", gather=lambda t: t),
                              _metrics={"train": defaultdict(list, {"mean_token_accuracy": [0.9, 0.8]})})
    callback.trainer = trainer
    callback.on_step_begin(None, None, "ctl")
    zeros = torch.zeros(3, dtype=torch.float64)
    spans = SimpleNamespace(count=zeros + torch.tensor([0.0, 10.0, 0.0], dtype=torch.float64),
                            correct=zeros + torch.tensor([0.0, 4.0, 0.0], dtype=torch.float64),
                            loss_sum=zeros + torch.tensor([0.0, 20.0, 0.0], dtype=torch.float64))
    meter.observe(spans)
    callback.on_step_end(None, SimpleNamespace(global_step=3), "ctl")
    assert trainer._metrics["train"]["mean_token_accuracy"] == [0.4]      # TRL 塞的两个数没了
    assert trainer._metrics["train"]["reasoning_loss"] == [2.0]
    assert "token_accuracy" not in trainer._metrics["train"]              # 不重名出两条
    assert "loss" not in trainer._metrics["train"]                        # train/loss 是 HF 自己那条


def test_span_collection_preserves_chunked_loss_and_backward_exactly(monkeypatch):
    """Multiple chunks and backward recomputation must preserve TRL's loss and gradients."""
    from training import span_metrics as metrics

    torch.set_num_threads(1)
    torch.manual_seed(71)
    assert metrics.install()
    hidden = torch.randn(1, 601, 8, requires_grad=True)
    weight = torch.randn(32, 8, requires_grad=True)
    bias = torch.randn(32, requires_grad=True)
    reference = [tensor.detach().clone().requires_grad_() for tensor in (hidden, weight, bias)]
    labels = torch.randint(0, 32, (1, 601))
    labels[:, :17] = -100
    labels[:, 90:96] = -100
    span = torch.where(labels == -100, 0, 2).to(torch.int8)
    span[:, 17:281] = torch.where(labels[:, 17:281] == -100, 0, 1)

    with monkeypatch.context() as patch:
        patch.setattr(metrics._sft, '_chunk', metrics._ORIGINAL_CHUNK)
        expected = metrics._ORIGINAL_CCE(reference[0], reference[1], 256,
                                         labels=labels, lm_head_bias=reference[2])
        expected[0].backward()
    with metrics.SpanMetrics.collect(span, labels) as collected:
        actual = metrics._sft._chunked_cross_entropy_loss(hidden, weight, 256,
                                                         labels=labels, lm_head_bias=bias)
    counts = collected.count.clone()
    actual[0].backward()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    for got, want in zip((hidden, weight, bias), reference):
        torch.testing.assert_close(got.grad, want.grad, rtol=0, atol=0)
    torch.testing.assert_close(collected.count, counts, rtol=0, atol=0)
    assert counts[1].item() == (span[:, 1:] == 1).sum().item()
    assert counts[2].item() == (span[:, 1:] == 2).sum().item()
