#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import re
from functools import partial

import mmcv
import numpy as np
from PIL import Image
from mmocr.utils.fileio import list_to_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training set of LSVT '
                    'by cropping box image.')
    parser.add_argument('root_path', help='Root dir path of LSVT')
    parser.add_argument(
        'n_proc', default=1, type=int, help='Number of processes to run')
    args = parser.parse_args()
    return args


def process_img(args, src_image_root, dst_image_root):
    # Dirty hack for multiprocessing
    img_idx, img_info, anns = args
    try:
        src_img = Image.open(osp.join(src_image_root, 'train_full_images_0/{}.jpg'.format(img_info)))
    except IOError:
        src_img = Image.open(osp.join(src_image_root, 'train_full_images_1/{}.jpg'.format(img_info)))
    blacklist = ['LOFTINESS*']
    whitelist = ['#Find YOUR Fun#', 'Story #', '*0#']
    labels = []
    for ann_idx, ann in enumerate(anns):
        text_label = ann['transcription']

        # Ignore illegible or words with non-Latin characters
        if ann['illegibility'] or re.findall(r'[\u4e00-\u9fff]+', text_label) or text_label in blacklist or \
                ('#' in text_label and text_label not in whitelist):
            continue

        points = np.asarray(ann['points'])
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)

        dst_img = src_img.crop((x1, y1, x2, y2))
        dst_img_name = f'img_{img_idx}_{ann_idx}.jpg'
        dst_img_path = osp.join(dst_image_root, dst_img_name)
        # Preserve JPEG quality
        dst_img.save(dst_img_path, qtables=src_img.quantization)
        labels.append(f'{osp.basename(dst_image_root)}/{dst_img_name}'
                      f' {text_label}')
    src_img.close()
    return labels


def convert_lsvt(root_path,
                 dst_image_path,
                 dst_label_filename,
                 annotation_filename,
                 img_start_idx=0,
                 nproc=1):
    annotation_path = osp.join(root_path, annotation_filename)
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')
    src_image_root = root_path

    # outputs
    dst_label_file = osp.join(root_path, dst_label_filename)
    dst_image_root = osp.join(root_path, dst_image_path)
    os.makedirs(dst_image_root, exist_ok=True)

    annotation = mmcv.load(annotation_path)

    process_img_with_path = partial(
        process_img,
        src_image_root=src_image_root,
        dst_image_root=dst_image_root)
    tasks = []
    for img_idx, (img_info, anns) in enumerate(annotation.items()):
        tasks.append((img_idx + img_start_idx, img_info, anns))
    labels_list = mmcv.track_parallel_progress(
        process_img_with_path, tasks, keep_order=True, nproc=nproc)
    final_labels = []
    for label_list in labels_list:
        final_labels += label_list
    list_to_file(dst_label_file, final_labels)
    return len(annotation)


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    convert_lsvt(
        root_path=root_path,
        dst_image_path='image_train',
        dst_label_filename='train_label.txt',
        annotation_filename='train_full_labels.json',
        nproc=args.n_proc)
    print('Finish')


if __name__ == '__main__':
    main()
