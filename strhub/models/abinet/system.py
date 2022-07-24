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

import logging
import math
from typing import Any, Tuple, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.optim.optim_factory import param_groups_weight_decay

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .model_abinet_iter import ABINetIterModel as Model

log = logging.getLogger(__name__)


class ABINet(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 iter_size: int, d_model: int, nhead: int, d_inner: int, dropout: float, activation: str,
                 v_loss_weight: float, v_attention: str, v_attention_mode: str, v_backbone: str, v_num_layers: int,
                 l_loss_weight: float, l_num_layers: int, l_detach: bool, l_use_self_attn: bool,
                 l_lr: float, a_loss_weight: float, lm_only: bool = False, **kwargs) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.scheduler = None
        self.save_hyperparameters()
        self.max_label_length = max_label_length
        self.num_classes = len(self.tokenizer) - 2  # We don't predict <bos> nor <pad>
        self.model = Model(max_label_length, self.eos_id, self.num_classes, iter_size, d_model, nhead, d_inner,
                           dropout, activation, v_loss_weight, v_attention, v_attention_mode, v_backbone, v_num_layers,
                           l_loss_weight, l_num_layers, l_detach, l_use_self_attn, a_loss_weight)
        self.model.apply(init_weights)
        # FIXME: doesn't support resumption from checkpoint yet
        self._reset_alignment = True
        self._reset_optimizers = True
        self.l_lr = l_lr
        self.lm_only = lm_only
        # Train LM only. Freeze other submodels.
        if lm_only:
            self.l_lr = lr  # for tuning
            self.model.vision.requires_grad_(False)
            self.model.alignment.requires_grad_(False)

    @property
    def _pretraining(self):
        # In the original work, VM was pretrained for 8 epochs while full model was trained for an additional 10 epochs.
        total_steps = self.trainer.estimated_stepping_batches * self.trainer.accumulate_grad_batches
        return self.global_step < (8 / (8 + 10)) * total_steps

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'model.language.proj.weight'}

    def _add_weight_decay(self, model: nn.Module, skip_list=()):
        if self.weight_decay:
            return param_groups_weight_decay(model, self.weight_decay, skip_list)
        else:
            return [{'params': model.parameters()}]

    def configure_optimizers(self):
        agb = self.trainer.accumulate_grad_batches
        # Linear scaling so that the effective learning rate is constant regardless of the number of GPUs used with DDP.
        lr_scale = agb * math.sqrt(self.trainer.num_devices) * self.batch_size / 256.
        lr = lr_scale * self.lr
        l_lr = lr_scale * self.l_lr
        params = []
        params.extend(self._add_weight_decay(self.model.vision))
        params.extend(self._add_weight_decay(self.model.alignment))
        # We use a different learning rate for the LM.
        for p in self._add_weight_decay(self.model.language, ('proj.weight',)):
            p['lr'] = l_lr
            params.append(p)
        max_lr = [p.get('lr', lr) for p in params]
        optim = AdamW(params, lr)
        self.scheduler = OneCycleLR(optim, max_lr, self.trainer.estimated_stepping_batches,
                                    pct_start=self.warmup_pct, cycle_momentum=False)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': self.scheduler, 'interval': 'step'}}

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        logits = self.model.forward(images)[0]['logits']
        return logits[:, :max_length + 1]  # truncate

    def calc_loss(self, targets, *res_lists) -> Tensor:
        total_loss = 0
        for res_list in res_lists:
            loss = 0
            if isinstance(res_list, dict):
                res_list = [res_list]
            for res in res_list:
                logits = res['logits'].flatten(end_dim=1)
                loss += F.cross_entropy(logits, targets.flatten(), ignore_index=self.pad_id)
            loss /= len(res_list)
            self.log('loss_' + res_list[0]['name'], loss)
            total_loss += res_list[0]['loss_weight'] * loss
        return total_loss

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if not self._pretraining and self._reset_optimizers:
            log.info('Pretraining ends. Updating base LRs.')
            self._reset_optimizers = False
            # Make base_lr the same for all groups
            base_lr = self.scheduler.base_lrs[0]  # base_lr of group 0 - VM
            self.scheduler.base_lrs = [base_lr] * len(self.scheduler.base_lrs)

    def _prepare_inputs_and_targets(self, labels):
        # Use dummy label to ensure sequence length is constant.
        dummy = ['0' * self.max_label_length]
        targets = self.tokenizer.encode(dummy + list(labels), self.device)[1:]
        targets = targets[:, 1:]  # remove <bos>. Unused here.
        # Inputs are padded with eos_id
        inputs = torch.where(targets == self.pad_id, self.eos_id, targets)
        inputs = F.one_hot(inputs, self.num_classes).float()
        lengths = torch.as_tensor(list(map(len, labels)), device=self.device) + 1  # +1 for eos
        return inputs, lengths, targets

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        inputs, lengths, targets = self._prepare_inputs_and_targets(labels)
        if self.lm_only:
            l_res = self.model.language(inputs, lengths)
            loss = self.calc_loss(targets, l_res)
        # Pretrain submodels independently first
        elif self._pretraining:
            # Vision
            v_res = self.model.vision(images)
            # Language
            l_res = self.model.language(inputs, lengths)
            # We also train the alignment model to 'satisfy' DDP requirements (all parameters should be used).
            # We'll reset its parameters prior to joint training.
            a_res = self.model.alignment(l_res['feature'].detach(), v_res['feature'].detach())
            loss = self.calc_loss(targets, v_res, l_res, a_res)
        else:
            # Reset alignment model's parameters once prior to full model training.
            if self._reset_alignment:
                log.info('Pretraining ends. Resetting alignment model.')
                self._reset_alignment = False
                self.model.alignment.apply(init_weights)
            all_a_res, all_l_res, v_res = self.model.forward(images)
            loss = self.calc_loss(targets, v_res, all_l_res, all_a_res)
        self.log('loss', loss)
        return loss

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        if self.lm_only:
            inputs, lengths, targets = self._prepare_inputs_and_targets(labels)
            l_res = self.model.language(inputs, lengths)
            loss = self.calc_loss(targets, l_res)
            loss_numel = (targets != self.pad_id).sum()
            return l_res['logits'], loss, loss_numel
        else:
            return super().forward_logits_loss(images, labels)
