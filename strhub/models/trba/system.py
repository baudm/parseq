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

from functools import partial
from typing import Sequence, Any, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply
from torch import Tensor

from strhub.models.base import CrossEntropySystem, CTCSystem
from strhub.models.utils import init_weights
from .model import TRBA as Model


class TRBA(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], num_fiducial: int, output_channel: int, hidden_size: int,
                 **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()
        self.max_label_length = max_label_length
        img_h, img_w = img_size
        self.model = Model(img_h, img_w, len(self.tokenizer), num_fiducial,
                           output_channel=output_channel, hidden_size=hidden_size, use_ctc=False)
        named_apply(partial(init_weights, exclude=['Transformation.LocalizationNetwork.localization_fc2']), self.model)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'model.Prediction.char_embeddings.weight'}

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        text = images.new_full([1], self.bos_id, dtype=torch.long)
        return self.model.forward(images, max_length, text)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        encoded = self.tokenizer.encode(labels, self.device)
        inputs = encoded[:, :-1]  # remove <eos>
        targets = encoded[:, 1:]  # remove <bos>
        max_length = encoded.shape[1] - 2  # exclude <bos> and <eos> from count
        logits = self.model.forward(images, max_length, inputs)
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        self.log('loss', loss)
        return loss


class TRBC(CTCSystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], num_fiducial: int, output_channel: int, hidden_size: int,
                 **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()
        self.max_label_length = max_label_length
        img_h, img_w = img_size
        self.model = Model(img_h, img_w, len(self.tokenizer), num_fiducial,
                           output_channel=output_channel, hidden_size=hidden_size, use_ctc=True)
        named_apply(partial(init_weights, exclude=['Transformation.LocalizationNetwork.localization_fc2']), self.model)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        # max_label_length is unused in CTC prediction
        return self.model.forward(images, None)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        loss = self.forward_logits_loss(images, labels)[1]
        self.log('loss', loss)
        return loss
