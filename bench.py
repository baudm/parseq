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

import os

import torch
from torch.utils import benchmark

from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table

import hydra
from omegaconf import DictConfig


@torch.inference_mode()
@hydra.main(config_path='configs', config_name='bench', version_base='1.2')
def main(config: DictConfig):
    # For consistent behavior
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    device = config.get('device', 'cuda')

    h, w = config.data.img_size
    x = torch.rand(1, 3, h, w, device=device)
    model = hydra.utils.instantiate(config.model).eval().to(device)

    if config.get('range', False):
        for i in range(1, 26, 4):
            timer = benchmark.Timer(
                stmt='model(x, len)',
                globals={'model': model, 'x': x, 'len': i})
            print(timer.blocked_autorange(min_run_time=1))
    else:
        timer = benchmark.Timer(
            stmt='model(x)',
            globals={'model': model, 'x': x})
        flops = FlopCountAnalysis(model, x)
        acts = ActivationCountAnalysis(model, x)
        print(timer.blocked_autorange(min_run_time=1))
        print(flop_count_table(flops, 1, acts, False))


if __name__ == '__main__':
    main()
