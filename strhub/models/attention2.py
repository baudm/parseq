"""
MHA, mostly from pytorch nn.MultiHeadAttention
"""
import warnings
from typing import Optional, Tuple, List
import math

import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn import Module
import torch.nn.functional as F


class MultiheadAttention(Module):
    """Mostly from nn.MultiheadAttention"""
    
    __constants__ = ['batch_first']

    def __init__(self, split_lengths:List[int], embed_dim, num_heads, dropout=0., batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.split_lengths = split_lengths
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        assert len(self.split_lengths) == 3, "only supports split_lengths of 3"

        self.in_proj_weight_1 = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_weight_2 = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_weight_3 = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))

        self.in_proj_bias_1 = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.in_proj_bias_2 = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.in_proj_bias_3 = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        
        self.out_proj_1 = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.out_proj_2 = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.out_proj_3 = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight_1)
        xavier_uniform_(self.in_proj_weight_2)
        xavier_uniform_(self.in_proj_weight_3)
        constant_(self.in_proj_bias_1, 0.)
        constant_(self.in_proj_bias_2, 0.)
        constant_(self.in_proj_bias_3, 0.)
        constant_(self.out_proj_1.bias, 0.)
        constant_(self.out_proj_2.bias, 0.)
        constant_(self.out_proj_3.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        assert query.shape[0] == sum(self.split_lengths), f"seq_len {query.shape[0]} does not match split_lengths {sum(self.split_lengths)}"
        
        attn_output, attn_output_weights = multi_head_attention_forward(
            self.split_lengths,
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight_1, self.in_proj_bias_1,
            self.in_proj_weight_2, self.in_proj_bias_2,
            self.in_proj_weight_3, self.in_proj_bias_3,
            self.dropout,
            self.out_proj_1.weight, self.out_proj_1.bias,
            self.out_proj_2.weight, self.out_proj_2.bias,
            self.out_proj_3.weight, self.out_proj_3.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
        

def multi_head_attention_forward(
    split_lengths: List[int],
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight_1: Tensor,
    in_proj_bias_1: Tensor,
    in_proj_weight_2: Tensor,
    in_proj_bias_2: Tensor,
    in_proj_weight_3: Tensor,
    in_proj_bias_3: Tensor,
    dropout_p: float,
    out_proj_weight_1: Tensor,
    out_proj_bias_1: Tensor,
    out_proj_weight_2: Tensor,
    out_proj_bias_2: Tensor,
    out_proj_weight_3: Tensor,
    out_proj_bias_3: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    average_attn_weights: bool = True
) -> Tuple[Tensor, Optional[Tensor]]:
    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
    
    # compute in-projection
    query1, query2, query3 = torch.split(query, split_lengths, dim=0)
    key1, key2, key3 = torch.split(key, split_lengths, dim=0)
    value1, value2, value3 = torch.split(value, split_lengths, dim=0)
    q1, k1, v1 = F._in_projection_packed(query1, key1, value1, in_proj_weight_1, in_proj_bias_1)
    q2, k2, v2 = F._in_projection_packed(query2, key2, value2, in_proj_weight_2, in_proj_bias_2)
    q3, k3, v3 = F._in_projection_packed(query3, key3, value3, in_proj_weight_3, in_proj_bias_3)
    if attn_mask is not None:
        assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
            f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # reshape q, k, v for multihead attention and make em batch first
    q1 = q1.contiguous().view(q1.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    q2 = q2.contiguous().view(q2.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    q3 = q3.contiguous().view(q3.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    k1 = k1.contiguous().view(k1.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    k2 = k2.contiguous().view(k2.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    k3 = k3.contiguous().view(k3.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v1 = v1.contiguous().view(v1.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v2 = v2.contiguous().view(v2.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v3 = v3.contiguous().view(v3.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    q = torch.cat([q1, q2, q3], dim=1)
    k = torch.cat([k1, k2, k3], dim=1)
    v = torch.cat([v1, v2, v3], dim=1)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # calculate attention and out projection
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous()
    attn_output1, attn_output2, attn_output3 = torch.split(attn_output, split_lengths, dim=0)
    attn_output1 = attn_output1.view(split_lengths[0] * bsz, embed_dim)
    attn_output2 = attn_output2.view(split_lengths[1] * bsz, embed_dim)
    attn_output3 = attn_output3.view(split_lengths[2] * bsz, embed_dim)
    attn_output1 = F.linear(attn_output1, out_proj_weight_1, out_proj_bias_1)
    attn_output2 = F.linear(attn_output2, out_proj_weight_2, out_proj_bias_2)
    attn_output3 = F.linear(attn_output3, out_proj_weight_3, out_proj_bias_3)
    attn_output1 = attn_output1.view(split_lengths[0], bsz, embed_dim)
    attn_output2 = attn_output2.view(split_lengths[1], bsz, embed_dim)
    attn_output3 = attn_output3.view(split_lengths[2], bsz, embed_dim)
    attn_output = torch.cat([attn_output1, attn_output2, attn_output3], dim=0)

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights
    else:
        return attn_output, None


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if attn_mask is not None:
        attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
    else:
        attn = torch.bmm(q, k.transpose(-2, -1))

    attn = F.softmax(attn, dim=-1)
    # remove nan
    attn = torch.nan_to_num(attn, 0)
    
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn