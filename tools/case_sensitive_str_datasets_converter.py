#!/usr/bin/env python3

import os.path
import sys
from pathlib import Path

d = sys.argv[1]
p = Path(d)

gt = []

num_samples = len(list(p.glob('label/*.txt')))
ext = 'jpg' if p.joinpath('IMG', '1.jpg').is_file() else 'png'

for i in range(1, num_samples + 1):
    img = p.joinpath('IMG', f'{i}.{ext}')
    name = os.path.splitext(img.name)[0]

    with open(p.joinpath('label', f'{i}.txt'), 'r') as f:
        label = f.readline()
    gt.append((os.path.join('IMG', img.name), label))

with open(d + '/lmdb.txt', 'w', encoding='utf-8') as f:
    for line in gt:
        fname, label = line
        fname = fname.strip()
        label = label.strip()
        f.write('\t'.join([fname, label]) + '\n')
