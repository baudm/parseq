#!/usr/bin/env python3
import argparse
import html
import math
import os
import os.path as osp
from functools import partial

import mmcv
from PIL import Image
from mmocr.utils.fileio import list_to_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of TextOCR '
                    'by cropping box image.')
    parser.add_argument('root_path', help='Root dir path of TextOCR')
    parser.add_argument(
        'n_proc', default=1, type=int, help='Number of processes to run')
    args = parser.parse_args()
    return args


def process_img(args, src_image_root, dst_image_root):
    # Dirty hack for multiprocessing
    img_idx, img_info, anns = args
    src_img = Image.open(osp.join(src_image_root, 'train2014', img_info['file_name']))
    src_w, src_h = src_img.size
    labels = []
    for ann_idx, ann in enumerate(anns):
        text_label = html.unescape(ann['utf8_string'].strip())

        # Ignore empty labels
        if not text_label or ann['class'] != 'machine printed' or ann['language'] != 'english' or \
                ann['legibility'] != 'legible':
            continue

        # Some labels and images with '#' in the middle are actually good, but some aren't, so we just filter them all.
        if text_label != '#' and '#' in text_label:
            continue

        # Some labels use '*' to denote unreadable characters
        if text_label.startswith('*') or text_label.endswith('*'):
            continue

        pad = 2
        x, y, w, h = ann['bbox']
        x, y = max(0, math.floor(x) - pad), max(0, math.floor(y) - pad)
        w, h = math.ceil(w), math.ceil(h)
        x2, y2 = min(src_w, x + w + 2 * pad), min(src_h, y + h + 2 * pad)
        dst_img = src_img.crop((x, y, x2, y2))
        dst_img_name = f'img_{img_idx}_{ann_idx}.jpg'
        dst_img_path = osp.join(dst_image_root, dst_img_name)
        # Preserve JPEG quality
        dst_img.save(dst_img_path, qtables=src_img.quantization)
        labels.append(f'{osp.basename(dst_image_root)}/{dst_img_name}'
                      f' {text_label}')
    src_img.close()
    return labels


def convert_textocr(root_path,
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
    split = 'train' if 'train' in dst_label_filename else 'val'

    process_img_with_path = partial(
        process_img,
        src_image_root=src_image_root,
        dst_image_root=dst_image_root)
    tasks = []
    for img_idx, img_info in enumerate(annotation['imgs'].values()):
        if img_info['set'] != split:
            continue
        ann_ids = annotation['imgToAnns'][str(img_info['id'])]
        anns = [annotation['anns'][str(ann_id)] for ann_id in ann_ids]
        tasks.append((img_idx + img_start_idx, img_info, anns))

    labels_list = mmcv.track_parallel_progress(
        process_img_with_path, tasks, keep_order=True, nproc=nproc)
    final_labels = []
    for label_list in labels_list:
        final_labels += label_list
    list_to_file(dst_label_file, final_labels)
    return len(annotation['imgs'])


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    num_train_imgs = convert_textocr(
        root_path=root_path,
        dst_image_path='image',
        dst_label_filename='train_label.txt',
        annotation_filename='cocotext.v2.json',
        nproc=args.n_proc)
    print('Processing validation set...')
    convert_textocr(
        root_path=root_path,
        dst_image_path='image_val',
        dst_label_filename='val_label.txt',
        annotation_filename='cocotext.v2.json',
        img_start_idx=num_train_imgs,
        nproc=args.n_proc)
    print('Finish')


if __name__ == '__main__':
    main()
