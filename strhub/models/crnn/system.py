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

from typing import Sequence, Optional

from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from strhub.models.base import CTCSystem
from strhub.models.utils import init_weights
from .model import CRNN as Model


class CRNN(CTCSystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], hidden_size: int, leaky_relu: bool, **kwargs) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()
        self.model = Model(img_size[0], 3, len(self.tokenizer), hidden_size, leaky_relu)
        self.model.apply(init_weights)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        return self.model.forward(images)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        loss = self.forward_logits_loss(images, labels)[1]
        self.log('loss', loss)
        return loss
