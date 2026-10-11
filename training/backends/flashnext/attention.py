"""Opt-in HF Qwen4Exp attention adapter; retain HF parameters, RoPE and gates.

Only one unpadded decision, without a KV cache, is supported. The Miles kernel
consumes O(T * budget) indices rather than a dense T * T attention mask.
This experimental adapter requires GPU forward/backward qualification.
"""

import math

import torch


def training_mask(*, inputs_embeds, attention_mask=None, past_key_values=None, **kwargs):
    if past_key_values is not None:
        raise ValueError("FlashNext SFT sparse attention requires use_cache=False")
    if inputs_embeds.shape[0] != 1:
        raise ValueError("FlashNext SFT sparse attention requires batch size 1")
    if attention_mask is not None and (
        attention_mask.shape != inputs_embeds.shape[:2] or not bool(attention_mask.all())
    ):
        raise ValueError("FlashNext SFT sparse attention requires an unpadded decision")
    positions = kwargs.get("position_ids")
    if positions is not None and bool((positions[..., 1:] < positions[..., :-1]).any()):
        raise ValueError("FlashNext SFT sparse attention does not support packed decisions")
    return attention_mask


@torch.no_grad()
def select_indices(indexer, hidden_states, position_embeddings, *, score_bytes=64 * 1024**2):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    if hidden_states.shape[0] != 1 or indexer.index_kv_heads != 1:
        raise ValueError("Expected one decision and one indexer KV head")
    _, length, _ = hidden_states.shape
    ratio, dim = indexer.compress_ratio, indexer.index_head_dim
    blocks = length // ratio
    width = min(blocks, indexer.block_topk)
    # Keep the incomplete causal tail as HF does, even when the budget is full.
    result = torch.full((length, width * ratio + ratio - 1), -1,
                        dtype=torch.int32, device=hidden_states.device)
    if blocks:
        qk = indexer.index_qk_proj(hidden_states)
        q, raw_keys = torch.split(qk, [indexer.index_n_heads * dim, dim], dim=-1)
        q = indexer.q_layernorm(q.view(1, length, indexer.index_n_heads, dim))
        cos, sin = position_embeddings
        q = apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=2)[0]
        pooled = raw_keys[:, :blocks * ratio].reshape(1, blocks, ratio, dim).float().mean(2)
        pooled = indexer.k_layernorm(pooled.to(raw_keys.dtype))
        starts = torch.arange(blocks, device=q.device) * ratio
        keys = apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos[:, starts], sin=sin[:, starts])[0, 0]
        block_ids = torch.arange(blocks, device=q.device)
        offsets = torch.arange(ratio, device=q.device)
        chunk = max(1, score_bytes // max(1, blocks * indexer.index_n_heads * 4))
        for start in range(0, length, chunk):
            end = min(length, start + chunk)
            scores = torch.relu(torch.einsum('qhd,kd->qhk', q[start:end].float(), keys.float()))
            scores = scores.sum(1) / math.sqrt(dim)
            visible = (torch.arange(start, end, device=q.device) + 1) // ratio
            scores.masked_fill_(block_ids[None] >= visible[:, None], -torch.inf)
            selected_scores, selected = scores.topk(width, dim=-1)
            # HF calls topk on the visible prefix. Resolve cutoff ties with
            # that same shape, since topk does not promise stable tie ordering.
            if width and width < blocks:
                cutoff = selected_scores[:, -1]
                tied = ((scores == cutoff[:, None]).sum(-1) >
                        (selected_scores == cutoff[:, None]).sum(-1)) & (visible > width)
                for row in tied.nonzero().flatten().tolist():
                    selected[row] = scores[row, :int(visible[row])].topk(width).indices
            tokens = (selected[..., None] * ratio + offsets).flatten(1)
            valid = (selected < visible[:, None])[..., None].expand(-1, -1, ratio).flatten(1)
            result[start:end, :width * ratio] = tokens.masked_fill(~valid, -1).int()
    positions = torch.arange(length, device=hidden_states.device)
    tail = ((positions + 1) // ratio * ratio)[:, None] + torch.arange(ratio - 1, device=positions.device)
    result[:, width * ratio:] = tail.masked_fill(tail > positions[:, None], -1).int()
    return result


def sparse_forward(self, hidden_states, position_embeddings, attention_mask=None,
                   past_key_values=None, **kwargs):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb
    from training.backends.flashnext.miles_qsa import qsa_sparse_attention

    training_mask(inputs_embeds=hidden_states, attention_mask=attention_mask,
                  past_key_values=past_key_values)
    if self.training and self.attention_dropout:
        raise ValueError("Miles QSA kernel does not implement attention dropout")
    indices = select_indices(self.indexer, hidden_states, position_embeddings)
    shape = hidden_states.shape[:-1]
    query, gate = self.q_proj(hidden_states).view(*shape, -1, self.head_dim * 2).chunk(2, dim=-1)
    query = self.q_norm(query).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(*shape, -1, self.head_dim)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(*shape, -1, self.head_dim)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    output = qsa_sparse_attention(query[0].transpose(0, 1), key[0].transpose(0, 1),
                                  value[0], indices, self.scaling)
    output = output.reshape(*shape, -1) * torch.sigmoid(gate.reshape(*shape, -1))
    return self.o_proj(output), None


def install():
    import transformers
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401

    if transformers.__version__ != "5.16.0":
        raise RuntimeError("FlashNext attention adapter currently requires transformers==5.16.0")
    # These symbols belong only to Qwen4Exp; other HF architectures are untouched.
    hf.create_causal_mask = training_mask
    hf.Qwen4ExpTextAttention.forward = sparse_forward
    # HF's existing fallback dispatcher resolves the local FLA package. Require
    # that dependency above, rather than silently permitting its Python loop.
