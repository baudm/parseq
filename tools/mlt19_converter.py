#!/usr/bin/env python3

import sys

root = sys.argv[1]

with open(root + '/gt.txt', 'r') as f:
    d = f.readlines()

with open(root + '/lmdb.txt', 'w') as f:
    for line in d:
        img, script, label = line.split(',', maxsplit=2)
        label = label.strip()
        if label and script in ['Latin', 'Symbols']:
            f.write('\t'.join([img, label]) + '\n')
