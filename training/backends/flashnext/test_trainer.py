"""Real tiny FlashNext VLM/PLE/MoE through our collator and TRL loss, on CPU.

Uses a dense reference in place of the CUDA kernel; the separate kernel tests
and full-size multi-GPU acceptance remain mandatory.
"""

import json
import inspect

import pytest
import torch

hf = pytest.importorskip('transformers.models.qwen4_exp.modeling_qwen4_exp')
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from datasets import Dataset
from trl import SFTConfig

from training.backends.flashnext import attention, miles_qsa
from training.backends.flashnext.test_attention import dense_kernel
from training.test_train_integration import examples, make_processor
from training.train_sft import FiniteSFTTrainer
from training.vlm_collator import VlmCollator


def test_flashnext_vlm_chunked_loss_trains_ple_and_moe(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    processor = make_processor()
    config = Qwen4ExpConfig(
        text_config=dict(vocab_size=len(processor.tokenizer), hidden_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, hc_count=4, hc_lowrank=8,
                         layer_types=['linear_attention', 'full_attention'],
                         linear_num_key_heads=2, linear_num_value_heads=4,
                         linear_key_head_dim=8, linear_value_head_dim=8,
                         moe_intermediate_size=16, shared_expert_intermediate_size=16,
                         num_experts=4, num_experts_per_tok=2, router_aux_loss_coef=0.,
                         ple_layer_ids=[1], ple_embed_dim=32, ngram_size=2, heads_per_ngram=2,
                         ngram_vocab_size_base=64, make_ngram_vocab_size_divisible_by=8,
                         split_ngram_parts=2, indexer_n_heads=2, indexer_kv_heads=1,
                         indexer_head_dim=8, indexer_budget=8, indexer_compress_ratio=4,
                         eos_token_id=processor.tokenizer.eos_token_id,
                         rope_parameters=dict(rope_type='default', rope_theta=10000.,
                                              partial_rotary_factor=1., mrope_section=[1, 1, 2]),
                         use_cache=False),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=32, num_position_embeddings=256, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2),
        **{name + '_token_id': processor.tokenizer.convert_tokens_to_ids(token)
           for name, token in [('image', '<|image_pad|>'), ('video', '<|video_pad|>'),
                               ('vision_start', '<|vision_start|>'), ('vision_end', '<|vision_end|>')]},
    )
    config._attn_implementation = 'sdpa'
    monkeypatch.setattr(hf, 'create_causal_mask', attention.training_mask)
    monkeypatch.setattr(hf.Qwen4ExpTextAttention, 'forward', attention.sparse_forward)
    monkeypatch.setattr(miles_qsa, 'qsa_sparse_attention', dense_kernel)
    # The installed CUDA packages otherwise take precedence even for CPU inputs.
    monkeypatch.setattr(hf, 'causal_conv1d_fn', inspect.unwrap(hf.causal_conv1d_fn))
    monkeypatch.setattr(hf, 'torch_chunk_gated_delta_rule', inspect.unwrap(hf.torch_chunk_gated_delta_rule))
    model = hf.Qwen4ExpForConditionalGeneration(config)
    rows = [dict(row_json=json.dumps(r), row_index=i) for i, r in enumerate(examples(tmp_path, 1))]
    collator = VlmCollator(processor, tmp_path, 1024)
    args = SFTConfig(output_dir=str(tmp_path / 'run'), use_cpu=True, bf16=False,
                     per_device_train_batch_size=1, per_device_eval_batch_size=1,
                     gradient_checkpointing=False, max_length=None, remove_unused_columns=False,
                     report_to='none', dataset_kwargs={'skip_prepare_dataset': True},
                     loss_type='chunked_nll')
    trainer = FiniteSFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                               processing_class=processor, data_collator=collator)
    batch = collator(rows)
    assert (batch['_sft_span'] == 1).any() and (batch['_sft_span'] == 2).any()
    loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    for fragment in ['ngram_embedding', '.experts.', '.visual.']:
        params = [p for name, p in model.named_parameters() if fragment in name]
        assert params, fragment
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params), fragment
    model.save_pretrained(tmp_path / 'saved')
    restored = hf.Qwen4ExpForConditionalGeneration.from_pretrained(tmp_path / 'saved')
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
