"""CPU semantic checks; GPU kernel checks run only with explicitly exposed CUDA."""

import copy
import inspect
import math

import pytest
import torch

hf = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from training.backends.flashnext.attention import select_indices, sparse_forward, training_mask


def fixture(length):
    torch.manual_seed(19)
    config = Qwen4ExpTextConfig(hidden_size=32, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=8,
                              indexer_n_heads=2, indexer_kv_heads=1,
                              indexer_head_dim=8, indexer_compress_ratio=4,
                              indexer_budget=8, attention_dropout=0.)
    config._attn_implementation = 'eager'
    layer = hf.Qwen4ExpTextAttention(config, layer_idx=0).float()
    hidden = torch.randn(1, length, 32)
    angles = torch.randn(1, length, 4)
    return layer, hidden, (angles.cos(), angles.sin())


@pytest.mark.parametrize('length', [1, 3, 4, 7, 8, 11, 25, 37])
def test_selection_matches_hf_including_partial_tail(length):
    layer, hidden, rope = fixture(length)
    causal = torch.ones(1, 1, length, length, dtype=torch.bool).tril()
    expected = layer.indexer(hidden, rope, causal, None)
    # Exercise several score chunks, not just a single full matrix.
    indices = select_indices(layer.indexer, hidden, rope, score_bytes=128)
    actual = torch.zeros(length, length + 1, dtype=torch.bool)
    actual.scatter_(1, indices.long().masked_fill(indices < 0, length), True)
    assert torch.equal(actual[:, :length], expected[0, 0])
    assert not torch.triu(actual[:, :length], diagonal=1).any()
    for row in indices:
        selected = row[row >= 0]
        assert len(selected) == len(selected.unique())


def dense_kernel(q, k, v, indices, scale):
    length = q.shape[0]
    mask = torch.zeros(length, length + 1, dtype=torch.bool, device=q.device)
    mask.scatter_(1, indices.long().masked_fill(indices < 0, length), True)
    repeat = q.shape[1] // k.shape[1]
    scores = torch.einsum('thd,shd->hts', q, k.repeat_interleave(repeat, 1)) * scale
    probs = scores.masked_fill(~mask[:, :length], -torch.inf).softmax(-1)
    return torch.einsum('hts,shd->thd', probs, v.repeat_interleave(repeat, 1))


def test_adapter_output_and_gradients_match_hf(monkeypatch):
    from training.backends.flashnext import miles_qsa
    monkeypatch.setattr(miles_qsa, 'qsa_sparse_attention', dense_kernel)
    reference, hidden, rope = fixture(25)
    adapter = copy.deepcopy(reference)
    x, y = hidden.requires_grad_(), hidden.clone().detach().requires_grad_()
    mask = torch.full((1, 1, 25, 25), -torch.inf).triu(1)
    expected = reference(x, rope, mask)[0]
    actual = sparse_forward(adapter, y, rope, torch.ones(1, 25))[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=3e-6, rtol=3e-5)
    for (name, p), (_, q) in zip(reference.named_parameters(), adapter.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5, msg=name)


def test_mask_rejects_padding_packing_cache_and_multiple_decisions():
    hidden = torch.zeros(1, 8, 32)
    assert training_mask(inputs_embeds=hidden) is None
    for kwargs in [dict(attention_mask=torch.zeros(1, 8)), dict(past_key_values=object()),
                   dict(position_ids=torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]))]:
        with pytest.raises(ValueError):
            training_mask(inputs_embeds=hidden, **kwargs)
    with pytest.raises(ValueError):
        training_mask(inputs_embeds=hidden.expand(2, -1, -1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires authorized CUDA device')
@pytest.mark.parametrize('length,budget', [(7, 8), (25, 8), (129, 8), (2177, 2048)])
def test_miles_gpu_kernel_forward_backward(length, budget):
    from training.backends.flashnext.miles_qsa import qsa_sparse_attention
    layer, hidden, rope = fixture(length)
    layer.indexer.block_topk = budget // layer.indexer.compress_ratio
    indices = select_indices(layer.indexer, hidden, rope).cuda()
    torch.manual_seed(21)
    inputs = [torch.randn(length, heads, 256, device='cuda', dtype=torch.bfloat16,
                          requires_grad=True) for heads in [24, 2, 2]]
    reference = [x.detach().float().requires_grad_() for x in inputs]
    expected = dense_kernel(*reference, indices, 1 / math.sqrt(256))
    actual = qsa_sparse_attention(*inputs, indices, 1 / math.sqrt(256))
    torch.testing.assert_close(actual.float(), expected, atol=.025, rtol=.025)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad.float())
    for x, y in zip(inputs, reference):
        torch.testing.assert_close(x.grad.float(), y.grad, atol=.06, rtol=.06)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires authorized CUDA device')
def test_fla_gpu_kernel_matches_hf_recurrence():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    torch.manual_seed(31)
    # Actual FlashNext head counts/dimensions after HF repeats the key heads.
    values = [torch.randn(1, 65, 48, 128, dtype=torch.bfloat16) for _ in range(3)]
    g = -torch.rand(1, 65, 48)
    beta = torch.rand(1, 65, 48, dtype=torch.bfloat16)
    reference = [x.float().requires_grad_() for x in values]
    inputs = [x.cuda().requires_grad_() for x in values]
    expected = inspect.unwrap(hf.torch_chunk_gated_delta_rule)(
        *reference, g=g, beta=beta.float(), use_qk_l2norm_in_kernel=True)[0]
    actual = chunk_gated_delta_rule(*inputs, g=g.cuda(), beta=beta.cuda(),
                                    use_qk_l2norm_in_kernel=True)[0]
    torch.testing.assert_close(actual.float().cpu(), expected, atol=.025, rtol=.04)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad.float().cpu())
    for x, y in zip(inputs, reference):
        torch.testing.assert_close(x.grad.float().cpu(), y.grad, atol=.04, rtol=.06)
