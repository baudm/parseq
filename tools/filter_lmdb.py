#!/usr/bin/env python3
import io
import os
from argparse import ArgumentParser

import numpy as np
import lmdb
from PIL import Image


def main():
    parser = ArgumentParser()
    parser.add_argument('inputs', nargs='+', help='Path to input LMDBs')
    parser.add_argument('--output', help='Path to output LMDB')
    parser.add_argument('--min_image_dim', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    with lmdb.open(args.output, map_size=1099511627776) as env_out:
        in_samples = 0
        out_samples = 0
        samples_per_chunk = 1000
        for lmdb_in in args.inputs:
            with lmdb.open(lmdb_in, readonly=True, max_readers=1, lock=False) as env_in:
                with env_in.begin() as txn:
                    num_samples = int(txn.get('num-samples'.encode()))
                in_samples += num_samples
                chunks = np.array_split(range(num_samples), num_samples // samples_per_chunk)
                for chunk in chunks:
                    cache = {}
                    with env_in.begin() as txn:
                        for index in chunk:
                            index += 1  # lmdb starts at 1
                            image_key = f'image-{index:09d}'.encode()
                            image_bin = txn.get(image_key)
                            img = Image.open(io.BytesIO(image_bin))
                            w, h = img.size
                            if w < args.min_image_dim or h < args.min_image_dim:
                                print(f'Skipping: {index}, w = {w}, h = {h}')
                                continue
                            out_samples += 1  # increment. start at 1
                            label_key = f'label-{index:09d}'.encode()
                            out_label_key = f'label-{out_samples:09d}'.encode()
                            out_image_key = f'image-{out_samples:09d}'.encode()
                            cache[out_label_key] = txn.get(label_key)
                            cache[out_image_key] = image_bin
                    with env_out.begin(write=True) as txn:
                        for k, v in cache.items():
                            txn.put(k, v)
                    print(f'Written samples from {chunk[0]} to {chunk[-1]}')
        with env_out.begin(write=True) as txn:
            txn.put('num-samples'.encode(), str(out_samples).encode())
        print(f'Written {out_samples} samples to {args.output} out of {in_samples} input samples.')


if __name__ == '__main__':
    main()
