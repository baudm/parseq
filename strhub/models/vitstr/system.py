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

from typing import Sequence, Any, Optional

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .model import ViTSTR as Model


class ViTSTR(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int, num_heads: int,
                 **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()
        self.max_label_length = max_label_length
        # We don't predict <bos> nor <pad>
        self.model = Model(img_size=img_size, patch_size=patch_size, depth=12, mlp_ratio=4, qkv_bias=True,
                           embed_dim=embed_dim, num_heads=num_heads, num_classes=len(self.tokenizer) - 2)
        # Non-zero weight init for the head
        self.model.head.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'model.' + n for n in self.model.no_weight_decay()}

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        logits = self.model.forward(images, max_length + 2)  # +2 tokens for [GO] and [s]
        # Truncate to conform to other models. [GO] in ViTSTR is actually used as the padding (therefore, ignored).
        # First position corresponds to the class token, which is unused and ignored in the original work.
        logits = logits[:, 1:]
        return logits

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        loss = self.forward_logits_loss(images, labels)[1]
        self.log('loss', loss)
        return loss
