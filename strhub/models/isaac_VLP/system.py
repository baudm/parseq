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
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .modules import DecoderLayer, Decoder, Encoder, TokenEmbedding

DEBUG_LAYER_INDEX = 0


class Isaac_VLP(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 dropout: float, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        print('Model : Isaac_VLP')
        self.save_hyperparameters()

        self.max_label_length = max_label_length

        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads,
                               mlp_ratio=enc_mlp_ratio)
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)

        # +1 for <eos>
        self.pos_embed = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        # attn_mask
        self.attn_mask = self.get_attn_mask(img_size, patch_size)
        
    def get_attn_mask(self, img_size, patch_size):
        """
        Creates attention mask for VLP Transformer.
        
        Attention between P & L are causal, including self.
        """
        L_V = int(img_size[0] * img_size[1] / (patch_size[0] * patch_size[1]))
        L_P = self.max_label_length + 1 # +1 for eos
        L_T = L_V + 2 * L_P
        attn_mask_PL = torch.triu(torch.full((L_P, L_P), float('-inf')), 1)
        attn_mask_PL = attn_mask_PL.repeat(2, 2)
        attn_mask = torch.zeros((L_T, L_T))
        attn_mask[-2 * L_P:, -2 * L_P:] = attn_mask_PL
        return attn_mask

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_embed'}
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(self, vis:torch.Tensor, lan:torch.Tensor,  pos:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        N, L = lan.shape
        
        # Add positional encoding to language tokens.
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(lan[:, :1])
        lan = self.pos_embed[:, :L - 1] + self.text_embed(lan[:, 1:])
        lan = self.dropout(torch.cat([null_ctx, lan], dim=1))
        
        if pos is None:
            pos = self.pos_embed[:, :L].expand(N, -1, -1)
        pos = self.dropout(pos)
        
        return self.decoder(vis, lan, pos, attn_mask=attn_mask)

    def forward(self, images:Tensor, max_length: Optional[int] = None) -> Tensor:
        """
        Forward-pass for val & test.
        
        Args:
            max_length: Max sequence length in batch, for efficient decoding in validation.
        """
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        L_P = num_steps = self.max_label_length + 1 # +1 for eos
        
        # prepare tokens
        vis = self.encode(images)
        lan = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
        lan[:, 0] = self.bos_id
        pos = self.pos_embed[:, :num_steps].expand(bs, -1, -1)
        
        attn_mask = self.attn_mask.to(self._device)
        
        logits = []
        for i in range(num_steps):
            j = i + 1 # next token index
            pos, _ = self.decode(vis, lan, pos, attn_mask=attn_mask)
            p_i = self.head(pos[:, i:j])
            logits.append(p_i)
            if j < num_steps:
                # greedy decode. add the next token index to the target input
                lan[:, j] = p_i.squeeze().argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if testing and (lan == self.eos_id).any(dim=-1).all():
                    break
        logits = torch.cat(logits, dim=1)
            
        return logits, None

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device)
        vis = self.encode(images)

        # Prepare the target sequences (input and output)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        vis, lan, pos, _ = self.decode(vis, tgt_in)
        logits = self.head(pos).flatten(end_dim=1)
        loss = F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)

        self.log('loss', loss)
        return loss
