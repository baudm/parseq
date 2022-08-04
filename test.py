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

import argparse
import string
import sys
from dataclasses import dataclass
from typing import List

import torch

from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)

    test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
    if args.new:
        test_set += SceneTextDataModule.TEST_NEW
    test_set = sorted(set(test_set))

    results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(model.device), labels), -1)['output']
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)

    result_groups = {
        'Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_SUB,
        'Benchmark': SceneTextDataModule.TEST_BENCHMARK
    }
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)


if __name__ == '__main__':
    main()
