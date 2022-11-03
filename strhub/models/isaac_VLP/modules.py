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

    def forward(self, vis, lan, pos, attn_mask:Optional[Tensor]=None):
        for i, dec_layer in enumerate(self.layers):
            vis, lan, pos, _ = dec_layer(vis, lan, pos, attn_mask)
        pos = self.norm(pos)
        return pos, None
    

class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_l = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_p = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)
        

    def forward(self, vis_tokens:Tensor, lan_tokens:Tensor, pos_tokens:Tensor, attn_mask:Optional[Tensor]=None):
        """
        Vision-Langauge-Position Transformer decoder.
        """
        L_V = vis_tokens.shape[1]
        L_L = lan_tokens.shape[1]
        L_P = pos_tokens.shape[1]
        
        tokens = torch.cat([vis_tokens, lan_tokens, pos_tokens], dim=1)
        lan_tokens_norm = self.norm_l(lan_tokens)
        pos_tokens_norm = self.norm_p(pos_tokens)
        tokens_norm = torch.cat([vis_tokens, lan_tokens_norm, pos_tokens_norm], dim=1)
        
        # SA
        tokens2, sa_weights = self.self_attn(tokens_norm, tokens_norm, tokens_norm, attn_mask=attn_mask)
        tokens = tokens + self.dropout1(tokens2)
        
        # FF
        tokens2 = self.linear2(self.dropout2(self.activation(self.linear1(tokens))))
        tokens = tokens + self.dropout3(tokens2)
        
        vis_tokens, lan_tokens, pos_tokens = torch.split(tokens, [L_V, L_L, L_P], dim=1)
        
        return vis_tokens, lan_tokens, pos_tokens, None