# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import transformer

from timm.models.vision_transformer import VisionTransformer, PatchEmbed

from strhub.models.attention import MultiheadAttention


@dataclass
class Module_Data:
    sa_weights: torch.Tensor=None


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class Encoder(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed):
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         num_classes=0, global_pool='', class_token=False)  # these disable the classifier head

    def forward(self, x):
        # Return all tokens
        return self.forward_features(x)
    

class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, vis, lan, pos, dummy, attn_mask:Optional[Tensor]=None, padding_mask:Optional[Tensor]=None):
        for i, dec_layer in enumerate(self.layers):
            if i + 1 == self.num_layers:
                vis_sec, lan_sec, pos_sec = vis, lan, pos
            vis, lan, pos, agg = dec_layer(vis, lan, pos, dummy, attn_mask, padding_mask)
            
        vis = self.norm(vis)
        lan = self.norm(lan)
        pos = self.norm(pos)
        return vis, lan, pos, vis_sec, lan_sec, pos_sec, agg
    

class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        # self.self_attn = MultiheadAttention([256, 26, 26], d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.ff = FeedForwardLayer(d_model, dim_feedforward, dropout, activation)
        
        self.norm_l = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_p = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.dummy_token = torch.zeros((1, d_model))
        

    def forward(self, vis_tokens:Tensor, lan_tokens:Tensor, pos_tokens:Tensor, dummy_token:Tensor,
                attn_mask:Optional[Tensor]=None, padding_mask:Optional[Tensor]=None):
        """
        Vision-Langauge-Position Transformer decoder.
        
        Dummy token is added to handle the softmax gradient error when all keys are masked.
        """
        L_V = vis_tokens.shape[1]
        L_L = lan_tokens.shape[1]
        L_P = pos_tokens.shape[1]
        
        tokens = torch.cat([vis_tokens, lan_tokens, pos_tokens, dummy_token], dim=1)
        tokens_norm = torch.cat([vis_tokens, self.norm_l(lan_tokens), self.norm_p(pos_tokens), dummy_token], dim=1)
        
        # SA
        tokens_res, sa_weights = self.self_attn(tokens_norm, tokens_norm, tokens_norm, attn_mask=attn_mask, key_padding_mask=padding_mask)
        tokens = tokens + self.dropout1(tokens_res)
        
        # FF
        tokens_res = self.ff(tokens)
        tokens = tokens + self.dropout2(tokens_res)
        vis_tokens, lan_tokens, pos_tokens, _ = torch.split(tokens, [L_V, L_L, L_P, 1], dim=1)
        
        # agg = Module_Data()
        # agg.sa_weights = sa_weights
        agg = None
        return vis_tokens, lan_tokens, pos_tokens, agg
    

class FeedForwardLayer(nn.Module):
    """Transformer position-wise feed-forward layer"""
    
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation='gelu', layer_norm_eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = transformer._get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(self.norm(x)))))