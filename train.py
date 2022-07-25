#!/usr/bin/env python3
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

from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem


@hydra.main(config_path='configs', config_name='train', version_base=None)
def main(config: DictConfig):
    trainer_strategy = None
    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        if config.trainer.get('resume_from_checkpoint', None) is not None:
            config.trainer.resume_from_checkpoint = hydra.utils.to_absolute_path(config.trainer.resume_from_checkpoint)
        # Special handling for GPU-affected config
        gpus = config.trainer.get('gpus', 0)
        if gpus:
            # Use mixed-precision training
            config.trainer.precision = 16
        if gpus > 1:
            # Use DDP
            config.trainer.strategy = 'ddp'
            # DDP optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            # Scale steps-based config
            config.trainer.val_check_interval //= gpus
            if config.trainer.get('max_steps', 0):
                config.trainer.max_steps //= gpus

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    model: BaseSystem = hydra.utils.instantiate(config.model)
    summarize(model, max_depth=1 if config.model.name.startswith('parseq') else 2)

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=3, save_last=True,
                                 filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}')
    swa = StochasticWeightAveraging(swa_epoch_start=0.75)
    cwd = Path.cwd()
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=TensorBoardLogger(str(cwd.parent), '', cwd.name),
                                               strategy=trainer_strategy, enable_model_summary=False,
                                               callbacks=[checkpoint, swa])
    ckpt_path = config.get('ckpt_path', None)
    if ckpt_path is not None:
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
