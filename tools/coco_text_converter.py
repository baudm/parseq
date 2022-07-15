#!/usr/bin/env python3

for s in ['train', 'val']:
    with open('{}_words_gt.txt'.format(s), 'r', encoding='utf8') as f:
        d = f.readlines()

    with open('{}_lmdb.txt'.format(s), 'w', encoding='utf8') as f:
        for line in d:
            try:
                fname, label = line.split(',', maxsplit=1)
            except ValueError:
                continue
            fname = '{}_words/{}.jpg'.format(s, fname.strip())
            label = label.strip().strip('|')
            f.write('\t'.join([fname, label]) + '\n')
