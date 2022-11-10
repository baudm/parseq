import os
import glob
import argparse
def str2bool(x):
    return x.lower() in ['true']
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='outputs', help='')
args, unknown = parser.parse_known_args()

for root, dirs, files in os.walk(args.root_dir):
    for file in files:
        if file.endswith('last.ckpt'):
            checkpoint_path = os.path.join(root, file)
            kwargs_str = ' '.join(unknown)
            print('#######################')
            print(f'python test.py {checkpoint_path} {kwargs_str}')
            print('#######################')
            os.system(f'python test.py {checkpoint_path} {kwargs_str}')