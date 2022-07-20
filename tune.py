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

import logging
import math
import os
import shutil
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.ray_trial_executor import RayTrialExecutor
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.suggest.ax import AxSearch

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem

log = logging.getLogger(__name__)


class MetricTracker(tune.Stopper):
    """Tracks the trend of the metric. Stops downward/stagnant trials. Assumes metric is being maximized."""

    def __init__(self, metric, max_t, patience: int = 3, window: int = 3) -> None:
        super().__init__()
        self.metric = metric
        self.trial_history = {}
        self.max_t = max_t
        self.training_iteration = 0
        self.eps = 0.01  # sensitivity
        self.patience = patience  # number of consecutive downward/stagnant samples to trigger early stoppage.
        self.kernel = self.gaussian_pdf(np.arange(window) - window // 2, sigma=0.6)
        # Extra samples to keep in order to have better MAs + gradients for the middle p samples.
        self.buffer = 2 * (len(self.kernel) // 2) + 2

    @staticmethod
    def gaussian_pdf(x, sigma=1.):
        return np.exp(-(x / sigma)**2 / 2) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def moving_average(x, k):
        return np.convolve(x, k, 'valid') / k.sum()

    def __call__(self, trial_id, result):
        self.training_iteration = result['training_iteration']
        if np.isnan(result['loss']) or self.training_iteration >= self.max_t:
            try:
                del self.trial_history[trial_id]
            except KeyError:
                pass
            return True
        history = self.trial_history.get(trial_id, [])
        # FIFO queue of metric values.
        history = history[-(self.patience + self.buffer - 1):] + [result[self.metric]]
        # Only start checking once we have enough data. At least one non-zero sample is required.
        if len(history) == self.patience + self.buffer and sum(history) > 0:
            smooth_grad = np.gradient(self.moving_average(history, self.kernel))[1:-1]  # discard edge values.
            # Check if trend is downward or stagnant
            if (smooth_grad < self.eps).all():
                log.info(f'Stopping trial = {trial_id}, hist = {history}, grad = {smooth_grad}')
                try:
                    del self.trial_history[trial_id]
                except KeyError:
                    pass
                return True
        self.trial_history[trial_id] = history
        return False

    def stop_all(self):
        return False


class TuneReportCheckpointPruneCallback(TuneReportCheckpointCallback):

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        self._checkpoint._handle(trainer, pl_module)
        # Prune older checkpoints
        for old in sorted(Path(tune.get_trial_dir()).glob('checkpoint_epoch=*-step=*'), key=os.path.getmtime)[:-1]:
            log.info(f'Deleting old checkpoint: {old}')
            shutil.rmtree(old)
        self._report._handle(trainer, pl_module)


def train(hparams, config, checkpoint_dir=None):
    with open_dict(config):
        config.model.lr = hparams['lr']
        # config.model.weight_decay = hparams['wd']
        if checkpoint_dir is not None:
            config.trainer.resume_from_checkpoint = os.path.join(checkpoint_dir, 'checkpoint')

    model: BaseSystem = hydra.utils.instantiate(config.model)
    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    tune_callback = TuneReportCheckpointPruneCallback({
        'loss': 'val_loss',
        'NED': 'val_NED',
        'accuracy': 'val_accuracy'
    })
    trainer: Trainer = hydra.utils.instantiate(config.trainer, progress_bar_refresh_rate=0, checkpoint_callback=False,
                                               logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name='',
                                                                        version='.'),
                                               callbacks=[tune_callback])
    trainer.fit(model, datamodule=datamodule)


@hydra.main(config_path='configs', config_name='tune')
def main(config: DictConfig):
    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'
    # Modify config
    with open_dict(config):
        # Use mixed-precision training
        if config.trainer.get('gpus', 0):
            config.trainer.precision = 16
        # We handle NaN here to terminate trials cleanly
        config.trainer.terminate_on_nan = False
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)

    tune_config = config.get('tune', {})
    lr = tune_config.get('lr', {})
    # wd = tune_config.get('wd', {})
    hparams = {
        'lr': tune.loguniform(lr.get('min', 1e-4), lr.get('max', 2e-3)),
        # 'wd': tune.loguniform(wd.get('min', 1e-4), wd.get('max', 1e-1)),
    }

    steps_per_epoch = len(hydra.utils.instantiate(config.data).train_dataloader())
    val_steps = steps_per_epoch * config.trainer.max_epochs / config.trainer.val_check_interval
    max_t = round(0.75 * val_steps)
    warmup_t = round(config.model.warmup_pct * val_steps)
    scheduler = MedianStoppingRule(time_attr='training_iteration', grace_period=warmup_t)

    # Always start by evenly diving the range in log scale.
    lr = hparams['lr']
    start = np.log10(lr.lower)
    stop = np.log10(lr.upper)
    num = math.ceil(stop - start) + 1
    initial_points = [{'lr': np.clip(x, lr.lower, lr.upper).item()} for x in reversed(np.logspace(start, stop, num))]
    search_alg = AxSearch(points_to_evaluate=initial_points)

    reporter = CLIReporter(
        parameter_columns=['lr'],
        metric_columns=['loss', 'accuracy', 'training_iteration'])

    analysis = tune.run(
        tune.with_parameters(train, config=config),
        name='trials',  # required. otherwise some of Ray's checks will fail.
        metric='NED',
        mode='max',
        stop=MetricTracker('NED', max_t),
        config=hparams,
        resources_per_trial={
            'cpu': 1,
            'gpu': tune_config.get('gpus_per_trial', 1)
        },
        num_samples=tune_config.get('num_samples', 10),
        local_dir=str(Path.cwd()),
        search_alg=search_alg,
        scheduler=scheduler,
        progress_reporter=reporter,
        resume=tune_config.get('resume', False),
        trial_executor=RayTrialExecutor(result_buffer_length=0)  # disable result buffering
    )

    print('Best hyperparameters found were: ', analysis.best_config)


if __name__ == '__main__':
    main()
