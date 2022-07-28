#!/usr/bin/env python3
import argparse
import string
import sys

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.abinet.system import ABINet

sys.path.insert(0, '.')
from hubconf import _get_config
from test import Result, print_results_table


class ABINetLM(ABINet):

    def _encode(self, labels):
        targets = [torch.arange(self.max_label_length + 1)]  # dummy target. used to set pad_sequence() length
        lengths = []
        for label in labels:
            targets.append(torch.as_tensor([self.tokenizer._stoi[c] for c in label]))
            lengths.append(len(label) + 1)
        targets = pad_sequence(targets, batch_first=True, padding_value=0)[1:]  # exclude dummy target
        lengths = torch.as_tensor(lengths, device=self.device)
        targets = F.one_hot(targets, len(self.tokenizer._stoi))[..., :len(self.tokenizer._stoi) - 2].float().to(self.device)
        return targets, lengths

    def forward(self, labels: Tensor, max_length: int = None) -> Tensor:
        targets, lengths = self._encode(labels)
        return self.model.language(targets, lengths)['logits']


def main():
    parser = argparse.ArgumentParser(description='Measure the word accuracy of ABINet LM using the ground truth as input')
    parser.add_argument('checkpoint', help='Official pretrained weights for ABINet-LV (best-train-abinet.pth)')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # charset used by original ABINet
    charset = string.ascii_lowercase + '1234567890'
    ckpt = torch.load(args.checkpoint)

    config = _get_config('abinet', charset_train=charset, charset_test=charset)
    model = ABINetLM(**config)
    model.model.load_state_dict(ckpt['model'])

    model = model.eval().to(args.device)
    model.freeze()  # disable autograd
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False)

    test_set = SceneTextDataModule.TEST_BENCHMARK
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
        for _, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((labels, labels), -1)['output']
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
        'Benchmark': SceneTextDataModule.TEST_BENCHMARK
    }
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    for group, subset in result_groups.items():
        print(f'{group} set:')
        print_results_table([results[s] for s in subset])
        print('\n')


if __name__ == '__main__':
    main()
