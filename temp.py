import lmdb
import glob
from pathlib import Path
from strhub.data.dataset import LmdbDataset

def do(root):
    for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
        mdb = Path(mdb)
        ds_name = str(mdb.parent.relative_to(root))
        ds_root = str(mdb.parent.absolute())
        num_samples = get_num_samples(ds_root)
        print(f'{ds_name} \t num samples: {num_samples}')

def get_num_samples(root):
    with lmdb.open(root, max_readers=1, readonly=True, create=False,
                         readahead=False, meminit=False, lock=False) as env, env.begin() as txn:
        return int(txn.get('num-samples'.encode()))

do(Path('../data/PARSeq/lmdb/'))