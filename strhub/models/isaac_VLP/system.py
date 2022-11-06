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

        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2) # We don't predict [B], [P]
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)

        # +1 for <eos>
        self.pos_embed = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        
        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        # attn_mask
        self.attn_mask = self.get_attn_mask(img_size, patch_size)
        self.visualize_attn_mask()
        self.dummy_token = torch.zeros((1, 1, embed_dim))
        
    # def get_attn_mask(self, img_size, patch_size):
    #     L_V = int(img_size[0] * img_size[1] / (patch_size[0] * patch_size[1]))
    #     L_L = L_P = self.max_label_length + 1 # +1 for eos
    #     L_T = L_V + L_L + L_P
    #     attn_mask_LP = torch.triu(torch.full((L_P, L_P), float('-inf')), 1)
    #     attn_mask_LP = attn_mask_LP.repeat(2, 2)
    #     attn_mask = torch.zeros((L_T, L_T))
    #     attn_mask[-L_L - L_P:, -L_L - L_P:] = attn_mask_LP
    #     return attn_mask
    
    def get_attn_mask(self, img_size, patch_size):
        L_V = int(img_size[0] * img_size[1] / (patch_size[0] * patch_size[1]))
        L_L = L_P = self.max_label_length + 1 # +1 for eos
        L_T = L_V + L_L + L_P
        def full_attn(h, w=None):
            w = w if w is not None else h
            return torch.zeros((h, w))
        def zero_attn(h, w=None):
            w = w if w is not None else h
            return torch.full((h, w), float('-inf'))
        def causal_attn(h, include_self=True):
            diagonal = 1 if include_self == True else 0
            return torch.triu(torch.full((h, h), float('-inf')), diagonal)
        def diag_attn(h):
            triu = torch.triu(torch.full((h, h), float('-inf'), 1))
            tril = torch.tril(torch.full((h, h), float('-inf'), -1))
            return triu + tril
        
        attn_VV = zero_attn(L_V)
        attn_VL = zero_attn(L_V, L_L)
        attn_VP = zero_attn(L_V, L_P)
        attn_V = torch.cat((attn_VV, attn_VL, attn_VP), dim=1)
        
        attn_LV = zero_attn(L_L, L_V)
        attn_LL = zero_attn(L_L)
        attn_LP = zero_attn(L_L, L_P)
        attn_L = torch.cat((attn_LV, attn_LL, attn_LP), dim=1)
        
        attn_PV = full_attn(L_P, L_V)
        attn_PL = zero_attn(L_P, L_L)
        attn_PP = zero_attn(L_P)
        attn_P = torch.cat((attn_PV, attn_PL, attn_PP), dim=1)
        
        attn_mask = torch.cat((attn_V, attn_L, attn_P), dim=0)
        attn_mask = self.add_dummy_attn(attn_mask)
        
        return attn_mask
    
    def add_dummy_attn(self, attn_mask):
        """ Add attention to dummy token (extra fixed zero token) to get around the
        gradient error caused by all keys being masked. When all keys are masked,
        attention to the dummy token is enabled.
        """
        attn_mask = F.pad(attn_mask, (0, 0, 0, 1), 'constant', float('-inf'))
        attn_mask = F.pad(attn_mask, (0, 1), 'constant', 0)
        for i, row in enumerate(attn_mask):
            if torch.any(row[:-1] != float('-inf')):
                attn_mask[i, -1] = float('-inf')
        return attn_mask

    def visualize_attn_mask(self):
        import seaborn as sns
        import pandas as pd
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        # L_L = L_P = self.max_label_length + 1
        # win = L_L + L_P
        win = self.attn_mask.shape[0]
        df = pd.DataFrame(torch.where(self.attn_mask == 0, 1, 0).numpy()[-win:, -win:], index=list(range(win)), columns=list(range(win)))
        s = 1.0
        plt.figure(figsize=(30 * s, 30 * s), dpi=300)
        annot_size = 10 * s
        tick_size = 5 * s
        labelsize = 15 * s
        save_path = f'./attn.png'
        ax = plt.gca()
        # ax_pos = [0.15, 0.01, 0.84, 0.84]
        # ax.set_position(ax_pos)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="5%")
        sa = sns.heatmap(df,
                        vmin=0,
                        vmax=1,
                        # annot=True,
                        # fmt='.2f',
                        # annot_kws={'size': annot_size},
                        ax=ax,
                        cbar_ax=cax,
                        cbar=True,
                        linewidths=0.5,
                        )
        cbar = sa.collections[0].colorbar
        cbar.ax.tick_params(labelsize=labelsize)
        sa.xaxis.tick_top()
        sa.set_xticklabels(sa.get_xmajorticklabels(), fontsize=tick_size, rotation=0)
        sa.set_yticklabels(sa.get_ymajorticklabels(), fontsize=tick_size, rotation=0)
        plt.savefig(save_path); plt.clf()

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_embed'}
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(self, vis:torch.Tensor, lan_ind:torch.Tensor,  pos:torch.Tensor, dummy_token:torch.Tensor,
               attn_mask:torch.Tensor, padding_mask:Optional[Tensor]=None):
        """
        Generate language / position tokens.
        Run Decoder.
        
        Args:
            vis : Visual tokens. Shape: N, L_V, D
            lan_ind : Language token indices. Shape: N, L_L
            pos : Positional tokens. Shape: N, L_P, D
        
        """
        # Add positional encoding to language tokens.
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        bs, L_L = lan_ind.shape
        null_ctx = self.text_embed(lan_ind[:, :1])
        lan = self.pos_embed[:, :L_L - 1] + self.text_embed(lan_ind[:, 1:])
        lan = self.dropout(torch.cat([null_ctx, lan], dim=1))
        
        pos = self.dropout(pos)
        
        dummy_token = dummy_token.expand(bs, -1, -1)
        
        return self.decoder(vis, lan, pos, dummy_token, attn_mask=attn_mask, padding_mask=padding_mask)

    def forward(self, images:Tensor, max_length: Optional[int] = None) -> Tensor:
        """
        Forward-pass for val & test.
        
        Args:
            max_length: Max sequence length in batch, for efficient decoding in validation.
        """
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        num_steps = max_length + 1 # +1 for eos
        L_L = L_P = self.max_label_length + 1 # +1 for eos
        
        # prepare tokens
        vis = self.encode(images)
        L_V = vis.shape[1]
        lan = torch.full((bs, L_L), self.pad_id, dtype=torch.long, device=self._device)
        lan[:, 0] = self.bos_id
        pos_in = self.pos_embed[:, :L_P].expand(bs, -1, -1)
        
        attn_mask = self.attn_mask.to(self._device)
        dummy_token = self.dummy_token.to(self._device)
        
        logits = []
        for i in range(num_steps):
            j = i + 1 # next token index
            pos_out, agg = self.decode(vis, lan, pos_in, dummy_token, attn_mask=attn_mask)
            p_i = self.head(pos_out[:, i:j])
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
        
        bs = images.shape[0]
        vis = self.encode(images)
        L_V = vis.shape[1]
        
        tgt = self.tokenizer.encode(labels, self._device)
        L_L = L_P = self.max_label_length + 1 # +1 for <eos>
        tgt = F.pad(tgt, (0, L_L + 1 - tgt.shape[1]), "constant", self.pad_id)
        # Prepare the target sequences (input and output)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        padding_mask = F.pad(padding_mask, (L_V, L_P + 1), "constant", 0) # +1 for dummy token
        
        pos = self.pos_embed[:, :L_P].expand(bs, -1, -1)
        
        attn_mask = self.attn_mask.to(self._device)
        dummy_token = self.dummy_token.to(self._device)
        
        pos, agg = self.decode(vis, tgt_in, pos, dummy_token, attn_mask, padding_mask)
        logits = self.head(pos).flatten(end_dim=1)
        loss = F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
        
        # if batch_idx % 100 == 0:
        #     pred = logits.argmax(-1).view(bs, -1)
        #     print('tgt_out')
        #     print(tgt_out)
        #     print('pred')
        #     print(pred)
            # chr_emb = self.text_embed(torch.LongTensor([0, 1, 2]).to(self._device))[:, :8]
            # print('chr_emb')
            # print(chr_emb)
            # pos_emb = self.pos_embed[0][:3][:, :8]
            # print('pos_emb')
            # print(pos_emb)
            # print('sa_weights')
            # print(agg.sa_weights[0][:5])
            # print(agg.sa_weights[0][-5:])
        
        self.log('loss', loss)
        
        return loss
