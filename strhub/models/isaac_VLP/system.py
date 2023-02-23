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
from typing import Sequence, Any, Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
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

@dataclass
class System_Data:
    sa_weights_dec: Tensor = None
    sa_weights_ref: Tensor = None

class Isaac_VLP(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int, ref_depth: int,
                 dropout: float, QK: List[List[str]], ref_loss_scale: int, ref_iters: int,
                 debug: bool = False, **kwargs: Any) -> None:
        """
        Args:
            QK : Specifies allowed attention. "VV" stands for self-attention of visual tokens.
                "PV" stands for positional tokens as query and visual tokens as key.
                
                QK = [query_V_list, query_L_list, query_P_list].
                query_V_list = [key_V, key_L, key_P]
                
                e.g. QK = [['V', 'L'], [], ['P']] means that "VV", "VL', "PP" attention is allowed.
                
                Language and positional tokens are always causal, including self.
        """
        self.debug = debug
        self.results_dec = []
        self.results_ref = []
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay, self.debug)
        print('Model : Isaac_VLP')
        self.save_hyperparameters()

        self.max_label_length = max_label_length

        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads,
                               mlp_ratio=enc_mlp_ratio)
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))
        self.refiner = Decoder(decoder_layer, num_layers=ref_depth, norm=nn.LayerNorm(embed_dim)) if ref_depth > 0 else None
        self.ref_loss_scale = ref_loss_scale
        self.ref_iters = ref_iters
        
        self.head = nn.Linear(embed_dim, len(self.tokenizer))
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)

        # +1 for <eos>
        self.pos_embed = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        
        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        # attn_mask
        self.QK = QK
        self.attn_mask = self.get_attn_mask(img_size, patch_size)
        self.attn_mask_refine = self.get_attn_mask(img_size, patch_size, refine_layer=True)
        self.visualize_attn_mask(self.attn_mask)
        self.visualize_attn_mask(self.attn_mask_refine, refine_layer=True)
        self.dummy_token = torch.zeros((1, 1, embed_dim))        
    
    def get_attn_mask(self, img_size, patch_size, refine_layer:bool=False):
        """Generates attention mask for the multi-modal transformer layers.
        
        Args:
            refine_layer: Whether or not the layer is used for refinement (as opposed to initial text prediction).
                When False, since information leak to future time steps are not allowed,
                - visual tokens cannot attend to language or positional tokens
                - causal mask is applied between language and positional tokens
                When True, it assumes an initial text prediction (up to <eos>) is already made.
                - full attention between visual, langauge and positional tokens is applied.
        """
        L_V = int(img_size[0] * img_size[1] / (patch_size[0] * patch_size[1]))
        L_L = L_P = self.max_label_length + 1 # +1 for eos
        L_T = L_V + L_L + L_P
        def full_attn(h, w=None):
            w = w if w is not None else h
            return torch.zeros((h, w))
        def zero_attn(h, w=None):
            w = w if w is not None else h
            return torch.full((h, w), float('-inf'))
        def causal_attn(h, w=None, include_self=True):
            w = w if w is not None else h
            diagonal = 1 if include_self == True else 0
            return torch.triu(torch.full((h, w), float('-inf')), diagonal)
        def diag_attn(h, w=None):
            w = w if w is not None else h
            triu = torch.triu(torch.full((h, w), float('-inf')), 1)
            tril = torch.tril(torch.full((h, w), float('-inf')), -1)
            return triu + tril
        def diag_mask(h, w=None, diagonal=0):
            w = w if w is not None else h
            base = torch.full((h, w), 1.0)
            triu = torch.triu(torch.full((h, w), -1.0), diagonal + 1)
            tril = torch.tril(torch.full((h, w), -1.0), diagonal - 1)
            mask = base + triu + tril
            mask = torch.zeros((h, w)).masked_fill(mask.type(torch.bool), float('-inf'))
            return mask
        
        # query : V
        QK_V = self.QK[0]
        if 'V' in QK_V:
            attn_VV = full_attn(L_V)
        else:
            attn_VV = zero_attn(L_V)
        if 'L' in QK_V and not not refine_layer:
            # VL attention is not allowed in base layer, due to information leak from future time steps
            attn_VL = full_attn(L_V, L_L)
        else:
            attn_VL = zero_attn(L_V, L_L)
        if 'P' in QK_V and not not refine_layer:
            # VP attention is not allowed in base layer, due to information leak from future time steps
            attn_VP = full_attn(L_V, L_P)
        else:
            attn_VP = zero_attn(L_V, L_P)
        attn_V = torch.cat((attn_VV, attn_VL, attn_VP), dim=1)
        
        # query : L
        QK_L = self.QK[1]
        if 'V' in QK_L:
            attn_LV = full_attn(L_L, L_V)
        else:
            attn_LV = zero_attn(L_L, L_V)
        if 'L' in QK_L:
            if not refine_layer:
                attn_LL = causal_attn(L_L)
            else:
                attn_LL = full_attn(L_L)
        else:
            attn_LL = zero_attn(L_L)
        if 'P' in QK_L:
            if not refine_layer:
                attn_LP = causal_attn(L_L, L_P)
            else:
                attn_LP = full_attn(L_L, L_P)
        else:
            attn_LP = zero_attn(L_L, L_P)
        attn_L = torch.cat((attn_LV, attn_LL, attn_LP), dim=1)
        
        # query : P
        QK_P = self.QK[2]
        if 'V' in QK_P:
            attn_PV = full_attn(L_P, L_V)
        else:
            attn_PV = zero_attn(L_P, L_V)
        if 'L' in QK_P:
            if not refine_layer:
                attn_PL = causal_attn(L_P, L_L)
            else:
                attn_PL = full_attn(L_P, L_L)
        else:
            attn_PL = zero_attn(L_P, L_L)
        if 'P' in QK_P:
            if not refine_layer:
                attn_PP = causal_attn(L_P)
            else:
                attn_PP = full_attn(L_P)
        else:
            attn_PP = zero_attn(L_P)
        attn_P = torch.cat((attn_PV, attn_PL, attn_PP), dim=1)
        
        attn_mask = torch.cat((attn_V, attn_L, attn_P), dim=0)
        attn_mask = self.add_dummy_attn(attn_mask)
        
        return attn_mask
    
    def add_dummy_attn(self, attn_mask):
        """ Add attention to dummy token(extra fixed zero token),
        which is appended to the end of the concatenated tokens, to get around the
        gradient error caused by all keys being masked. When all keys are masked,
        attention to the dummy token is enabled.
        """
        attn_mask = F.pad(attn_mask, (0, 0, 0, 1), 'constant', float('-inf'))
        attn_mask = F.pad(attn_mask, (0, 1), 'constant', 0)
        for i, row in enumerate(attn_mask):
            if torch.any(row[:-1] != float('-inf')):
                attn_mask[i, -1] = float('-inf')
        return attn_mask

    def visualize_attn_mask(self, attn_mask, refine_layer:bool=False):
        import seaborn as sns
        import pandas as pd
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        vis_size = [a // b for (a, b) in zip(self.hparams.img_size, self.hparams.patch_size)]
        L_V = vis_size[0] * vis_size[1]
        L_L = L_P = self.max_label_length + 1
        L_T = L_V + L_L + L_P
        win = attn_mask.shape[0]
        df = pd.DataFrame(torch.where(attn_mask == 0, 1, 0).numpy()[-win:, -win:], index=list(range(win)), columns=list(range(win)))
        s = 1.0
        plt.figure(figsize=(30 * s, 30 * s), dpi=300)
        annot_size = 10 * s
        tick_size = 5 * s
        labelsize = 15 * s
        if refine_layer:
            save_path = f'./attn_refine.png'
        else:
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
        rects = []
        for x, y, w, h in [(0, 0, L_V, L_V), (L_V, 0, L_L, L_V), (L_V + L_L, 0, L_P, L_V), (L_T, 0, 1, L_V),
         (0, L_V, L_V, L_L), (L_V, L_V, L_L, L_L), (L_V + L_L, L_V, L_P, L_L), (L_T, L_V, 1, L_L),
         (0, L_V + L_L, L_V, L_P), (L_V, L_V + L_L, L_L, L_P), (L_V + L_L, L_V + L_L, L_P, L_P), (L_T, L_V + L_L, 1, L_P),
         (0, L_T, L_V, 1), (L_V, L_T, L_L, 1), (L_V + L_L, L_T, L_P, 1), (L_T, L_T, 1, 1),
         ]:
            rects.append(patches.Rectangle((x, y,), w, h, edgecolor='green', facecolor='none', linewidth=3))
        for rect in rects:
            sa.add_patch(rect)
        sa.xaxis.tick_top()
        sa.set_xticklabels(sa.get_xmajorticklabels(), fontsize=tick_size, rotation=0)
        sa.set_yticklabels(sa.get_ymajorticklabels(), fontsize=tick_size, rotation=0)
        plt.savefig(save_path); plt.clf()

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_embed'}
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)
    
    def forward(self, images:Tensor, validation: bool = False, debug: bool = False) -> Tensor:
        """
        Forward-pass for test & val.
        Used implicitly in forward-pass of val.
        
        Args:
            images : input image
            return_intermediate_logits : In case of decoder-refiner structure, also return decoder logits.
            validation : Is validation step.
            debug : Is debug mode.
        """
        if debug :
            agg_system = System_Data()
        else:
            agg_system = None
        
        testing = not validation
        bs = images.shape[0]
        num_steps = self.max_label_length + 1 # +1 for eos
        L_L = L_P = num_steps = self.max_label_length + 1 # +1 for eos
        
        #@ decoder
        #* prepare tokens
        vis = self.encode(images)
        L_V = vis.shape[1]
        lan_ind = torch.full((bs, L_L), self.pad_id, dtype=torch.long, device=self._device)
        lan_ind[:, 0] = self.bos_id
        pos = self.pos_embed[:, :L_P].expand(bs, -1, -1)
        dummy_token = self.dummy_token.to(self._device)
        attn_mask = self.attn_mask.to(self._device)
        #* decoding
        logits_dec = []
        agg_dec_ts = []
        for i in range(num_steps):
            j = i + 1 # next token index
            lan = self.to_lan(lan_ind)
            vis_dec, lan_dec, pos_dec, vis_dec_2, lan_dec_2, pos_dec_2, agg_dec_t = self.decode(vis, lan, pos, dummy_token, attn_mask=attn_mask, debug=debug)
            agg_dec_ts.append(agg_dec_t)
            p_i = self.head(pos_dec[:, i:j])
            logits_dec.append(p_i)
            max_time_step = i
            if j < num_steps:
                # greedy decode. add the next token index to the target input
                lan_ind[:, j] = p_i.squeeze().argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if testing and (lan_ind == self.eos_id).any(dim=-1).all():
                    break
        logits_dec = torch.cat(logits_dec, dim=1)
        logits = logits_dec
        
        if debug:
            DEC_IDX = 0
            sa_weights = []
            for t in range(max_time_step + 1):
                _sa_weights = agg_dec_ts[t][DEC_IDX].sa_weights[0]
                sa_weights.append(_sa_weights)
            sa_weights = torch.stack(sa_weights)
            agg_system.sa_weights_dec = sa_weights
        
        #@ refiner
        if self.refiner is not None and self.ref_iters > 0:
            #* lan tokens
            # remove eos token
            init_pred_ind = logits_dec.argmax(-1)[:, :-1]
            # prepend bos token
            bos = torch.full((logits_dec.shape[0], 1), self.bos_id).to(self._device)
            init_pred_ind = torch.cat([bos, init_pred_ind], dim=1)
            #- convert eos token to pad / pad up to input size
            # convert eos position and beyond to pad
            pad_positions = ((init_pred_ind == self.eos_id).cumsum(-1) > 0).to(torch.bool)
            init_pred_ind = init_pred_ind.masked_fill(pad_positions, self.pad_id)
            init_pred_ind = F.pad(init_pred_ind, (0, L_L - init_pred_ind.shape[1]), "constant", self.pad_id)
            padding_mask = (init_pred_ind == self.pad_id)
            # pad up to input size
            padding_mask = F.pad(padding_mask, (L_V, L_P + 1), "constant", 0) # +1 for dummy token
            init_pred_lan = self.to_lan(init_pred_ind)
            #* attention mask
            attn_mask_refine = self.attn_mask_refine.to(self._device)
            #* refine
            vis_ref, lan_ref, pos_ref, vis_ref_2, lan_ref_2, pos_ref_2, agg_ref = self.refine(vis_dec_2, init_pred_lan, pos_dec, dummy_token, attn_mask_refine, padding_mask, debug=debug)
            logits_ref = self.head(pos_ref)
            logits = logits_ref
            
            if debug:
                REF_IDX = 0
                sa_weights = agg_ref[REF_IDX].sa_weights[0].unsqueeze(0)
                agg_system.sa_weights_ref = sa_weights
        
        return logits, logits_dec, agg_system
    
    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        """
        Forward-pass for val.
        Override function defined in CrossEntropySystem, because initial prediction might be longer than target."""
        logits, logits_inter, _ = self.forward(images, validation=True)
        tokens = self.tokenizer.encode(labels, self.device)
        tgt_out = tokens[:, 1:]  # Discard <bos>
        L_L = self.max_label_length + 1 # +1 for <eos>
        tgt_out = F.pad(tgt_out, (0, L_L - tgt_out.shape[1]), "constant", self.pad_id)
        loss = F.cross_entropy(logits.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)
        loss_inter = F.cross_entropy(logits_inter.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)
        loss_numel = (tgt_out != self.pad_id).sum()
        return logits, loss, logits_inter, loss_inter, loss_numel

    def encode(self, img: torch.Tensor):
        return self.encoder(img)
    
    def to_lan(self, tgt_in):
        bs, L_L = tgt_in.shape
        null_ctx = self.text_embed(tgt_in[:, :1]) # tgt_in stats with [B]. No positional encoding added to [B]
        lan = torch.cat([null_ctx, self.text_embed(tgt_in[:, 1:]) + self.pos_embed[:, :L_L - 1]], dim=1)
        return lan

    def decode(self, vis:torch.Tensor, lan:torch.Tensor,  pos:torch.Tensor, dummy_token:torch.Tensor,
               attn_mask:torch.Tensor, padding_mask:Optional[Tensor]=None, debug=False):
        """
        Used in forward-pass of train.
        Run Decoder.
        
        Args:
            vis : Visual tokens. Shape: N, L_V, D
            lan : Language tokens. Shape: N, L_L, D
            pos : Positional tokens. Shape: N, L_P, D
        
        """
        lan = self.dropout(lan)
        pos = self.dropout(pos)
        dummy_token = dummy_token.expand(pos.shape[0], -1, -1)
        return self.decoder(vis, lan, pos, dummy_token, attn_mask=attn_mask, padding_mask=padding_mask, debug=debug)
    
    def refine(self, vis:torch.Tensor, lan:torch.Tensor,  pos:torch.Tensor, dummy_token:torch.Tensor,
               attn_mask:torch.Tensor, padding_mask:Optional[Tensor]=None, debug=False):
        """
        Used in forward-pass of train.
        Further refines initial decoder prediction.
        Stop gradient is applied to language and positional tokens,
        to prevent information leak from future steps.
        
        Args:
            vis : Visual tokens. Shape: N, L_V, D
            lan : Language tokens. Shape: N, L_L, D
            pos : Positional tokens. Shape: N, L_P, D
        
        """
        lan = self.dropout(lan)
        pos = self.dropout(pos)
        dummy_token = dummy_token.expand(pos.shape[0], -1, -1)
        return self.refiner(vis, lan.detach(), pos.detach(), dummy_token, attn_mask=attn_mask, padding_mask=padding_mask, debug=debug)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        bs = images.shape[0]
        
        #* vis tokens
        vis = self.encode(images)
        L_V = vis.shape[1]
        
        #@ decoding stage.
        #* lan tokens
        tokens = self.tokenizer.encode(labels, self._device)
        L_L = L_P = self.max_label_length + 1 # +1 for <eos>
        tokens = F.pad(tokens, (0, L_L + 1 - tokens.shape[1]), "constant", self.pad_id) # +1 for <bos>
        tgt_in = tokens[:, :-1]
        tgt_out = tokens[:, 1:]
        # padding mask : pad + eos posiitons
        padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        padding_mask = F.pad(padding_mask, (L_V, L_P + 1), "constant", 0) # +1 for dummy token
        lan = self.to_lan(tgt_in)
        #* pos tokens
        pos = self.pos_embed[:, :L_P].expand(bs, -1, -1)
        #* dummy token
        dummy_token = self.dummy_token.to(self._device)
        #* attention mask
        attn_mask = self.attn_mask.to(self._device)
        #* decoding
        vis_dec, lan_dec, pos_dec, vis_dec_2, lan_dec_2, pos_dec_2, agg = self.decode(vis, lan, pos, dummy_token, attn_mask, padding_mask)
        logits_dec = self.head(pos_dec)
        loss_dec = F.cross_entropy(logits_dec.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)
        probs_dec = logits_dec.softmax(-1)
        preds_dec, probs_dec = self.tokenizer.decode(probs_dec)
        results_dec = [(a == b) for (a, b) in zip(labels, preds_dec)]
        self.results_dec.extend(results_dec)
        if len(self.results_dec) > 10000:
            train_acc_dec = sum(self.results_dec) / len(self.results_dec) * 100
            self.log('train_acc_dec', train_acc_dec)
            self.results_dec = []
        
        #@ refinement stage.
        if self.refiner is not None:
            #* lan tokens
            bos = torch.full((bs, 1), self.bos_id).to(self._device)
            # remove eos token
            init_pred_ind = logits_dec.argmax(-1)[:, :-1]
            # prepend bos token
            init_pred_ind = torch.cat([bos, init_pred_ind], dim=1)
            #- convert eos token to pad / pad up to input size
            # convert eos position and beyond to pad
            pad_positions = ((init_pred_ind == self.eos_id).cumsum(-1) > 0).to(torch.bool)
            init_pred_ind = init_pred_ind.masked_fill(pad_positions, self.pad_id)
            init_pred_ind = F.pad(init_pred_ind, (0, L_L - init_pred_ind.shape[1]), "constant", self.pad_id)
            padding_mask = (init_pred_ind == self.pad_id)
            # pad up to input size
            padding_mask = F.pad(padding_mask, (L_V, L_P + 1), "constant", 0) # +1 for dummy token
            init_pred_lan = self.to_lan(init_pred_ind)
            #* attention mask
            attn_mask_refine = self.attn_mask_refine.to(self._device)
            #* refiner
            vis_ref, lan_ref, pos_ref, vis_ref_2, lan_ref_2, pos_ref_2, agg = self.refine(vis_dec_2, init_pred_lan, pos_dec, dummy_token, attn_mask_refine, padding_mask)
            #- loss
            logits_ref = self.head(pos_ref)
            loss_ref = F.cross_entropy(logits_ref.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)
            loss_ref = self.ref_loss_scale * loss_ref
            loss = loss_dec + loss_ref
            #- accuracy
            probs_ref = logits_ref.softmax(-1)
            preds_ref, probs_ref = self.tokenizer.decode(probs_ref)
            results_ref = [(a == b) for (a, b) in zip(labels, preds_ref)]
            self.results_ref.extend(results_ref)
            if len(self.results_ref) > 10000:
                train_acc_ref = sum(self.results_ref) / len(self.results_ref) * 100
                self.log('train_acc_ref', train_acc_ref)
                self.results_ref = []
        else:
            loss_ref = 0
            loss = loss_dec
        
        self.log('loss', loss)
        self.log('loss_ref', loss_ref)
        self.log('loss_dec', loss_dec)
        
        
        return loss
