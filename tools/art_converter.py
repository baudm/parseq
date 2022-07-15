#!/usr/bin/env python3

import json

with open('train_task2_labels.json', 'r', encoding='utf8') as f:
    d = json.load(f)

with open('gt.txt', 'w', encoding='utf8') as f:
    for k, v in d.items():
        if len(v) != 1:
            print('error', v)
        v = v[0]
        if v['language'].lower() != 'latin':
            # print('Skipping non-Latin:', v)
            continue
        if v['illegibility']:
            # print('Skipping unreadable:', v)
            continue
        label = v['transcription'].strip()
        if not label:
            # print('Skipping blank label')
            continue
        if '#' in label and label != 'LocaL#3':
            # print('Skipping corrupted label')
            continue
        f.write('\t'.join(['train_task2_images/' + k + '.jpg', label]) + '\n')
