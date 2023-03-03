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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from nltk import edit_distance
from timm.optim import create_optimizer_v2
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR

from strhub.data.utils import CharsetAdapter, Tokenizer, BaseTokenizer


@dataclass
class BatchResult:
    num_samples: int
    correct: int
    ned: float
    confidence: float
    label_length: int
    loss: Tensor
    loss_numel: int


class BaseSystem(pl.LightningModule, ABC):

    def __init__(self, tokenizer: BaseTokenizer, charset_test: str,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 debug: bool = False) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.charset_adapter = CharsetAdapter(charset_test)
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_pct = warmup_pct
        self.weight_decay = weight_decay
        self.debug = debug

    @abstractmethod
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        """Inference

        Args:
            images: Batch of images. Shape: N, Ch, H, W
            max_length: Max sequence length of the output. If None, will use default.

        Returns:
            logits: N, L, C (L = sequence length, C = number of classes, typically len(charset_train) + num specials)
        """
        raise NotImplementedError

    @abstractmethod
    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        """Like forward(), but also computes the loss (calls forward() internally).

        Args:
            images: Batch of images. Shape: N, Ch, H, W
            labels: Text labels of the images

        Returns:
            logits: N, L, C (L = sequence length, C = number of classes, typically len(charset_train) + num specials)
            loss: mean loss for the batch
            loss_numel: number of elements the loss was calculated from
        """
        raise NotImplementedError

    def configure_optimizers(self):
        # Linear scaling so that the effective learning rate is constant regardless of the number of GPUs used with DDP.
        agb = self.trainer.accumulate_grad_batches # 1
        lr_scale = agb * math.sqrt(self.trainer.num_devices) * self.batch_size / 256.
        lr = lr_scale * self.lr
        optim = create_optimizer_v2(self, 'adamw', lr, self.weight_decay)
        sched = OneCycleLR(optim, lr, self.trainer.estimated_stepping_batches, pct_start=self.warmup_pct,
                           cycle_momentum=False)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
        optimizer.zero_grad(set_to_none=True)

    def _eval_step(self, batch, validation: bool, debug_dir='', dname='') -> Optional[STEP_OUTPUT]:
        if self.debug:
            images, labels, img_keys, img_origs = batch
        else:
            images, labels = batch
            img_keys = [None] * len(images)
            img_origs = [None] * len(images)

        total, correct, ned, confidence, label_length = 0, 0, 0, 0, 0
        correct_inter, ned_inter, confidence_inter = 0, 0, 0
        if validation:
            logits, loss, logits_inter, loss_inter, loss_numel = self.forward_logits_loss(images, labels)
        else:
            # if 'traditional' in labels:
            #     print(labels.index('traditional'))
            #     import ipdb; ipdb.set_trace(context=11) # #FF0000
            logits, logits_inter, agg = self.forward(images)
            loss = loss_inter = loss_numel = None

        probs = logits.softmax(-1)
        preds, probs = self.tokenizer.decode(probs)
        preds = [self.charset_adapter(pred) for pred in preds]
        
        for pred, prob, gt, img_key, img_orig in zip(preds, probs, labels, img_keys, img_origs):
            confidence += prob.prod().item()
            ned += edit_distance(pred, gt) / max(len(pred), len(gt))
            if pred == gt:
                correct += 1
            else:
                if self.debug:
                    # Save error images
                    img_orig.save(f'{debug_dir}/images/{dname}/{img_key}_{gt.replace("/", chr(0x2215))}_{pred.replace("/", chr(0x2215))}.png')
            total += 1
            label_length += len(pred)
            
        probs_inter = logits_inter.softmax(-1)
        preds_inter, probs_inter = self.tokenizer.decode(probs_inter)
        preds_inter = [self.charset_adapter(pred_inter) for pred_inter in preds_inter]
        
        for pred_inter, prob_inter, gt, img_key, img_orig in zip(preds_inter, probs_inter, labels, img_keys, img_origs):
            confidence_inter += prob_inter.prod().item()
            ned_inter += edit_distance(pred_inter, gt) / max(len(pred_inter), len(gt))
            if pred_inter == gt:
                correct_inter += 1
            else:
                if self.debug:
                    pass
        
        result = dict(output=BatchResult(total, correct, ned, confidence, label_length, loss, loss_numel),
                    output_inter=BatchResult(total, correct_inter, ned_inter, confidence_inter, label_length, loss_inter, loss_numel),
                    preds = preds, preds_inter=preds_inter)
        
        return result

    @staticmethod
    def _aggregate_results(outputs: EPOCH_OUTPUT) -> Tuple[float, float, float, float]:
        if not outputs:
            return 0., 0., 0.
        total_size = 0
        total_size_inter = 0
        total_loss_numel = 0
        total_loss_numel_inter = 0
        total_loss = 0
        total_loss_inter = 0
        total_n_correct = 0
        total_n_correct_inter = 0
        for result in outputs:
            total_size += result['output'].num_samples
            total_loss_numel += result['output'].loss_numel
            total_loss += result['output'].loss_numel * result['output'].loss
            total_n_correct += result['output'].correct
        acc = total_n_correct / total_size
        loss = total_loss / total_loss_numel
        for result in outputs:
            total_size_inter += result['output_inter'].num_samples
            total_loss_numel_inter += result['output_inter'].loss_numel
            total_loss_inter += result['output_inter'].loss_numel * result['output_inter'].loss
            total_n_correct_inter += result['output_inter'].correct
        acc_inter = total_n_correct_inter / total_size_inter
        loss_inter = total_loss_inter / total_loss_numel_inter
        return acc, loss, acc_inter, loss_inter

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self._eval_step(batch, True)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        acc, loss, acc_inter, loss_inter = self._aggregate_results(outputs)
        self.log('val_accuracy', 100 * acc, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_accuracy_inter', 100 * acc_inter, sync_dist=True)
        self.log('val_loss_inter', loss_inter, sync_dist=True)
        self.log('hp_metric', acc, sync_dist=True)

    def test_step(self, batch, batch_idx, debug_dir='', dname='') -> Optional[STEP_OUTPUT]:
        return self._eval_step(batch, False, debug_dir, dname)


class CrossEntropySystem(BaseSystem):

    def __init__(self, charset_train: str, charset_test: str,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 debug: bool = False) -> None:
        tokenizer = Tokenizer(charset_train)
        self.debug = debug
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay, self.debug)
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        logits, _, _ = self.forward(images, max_len)
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, logits, loss, loss_numel