from argparse import ArgumentParser
import os
import glob
import random
import re
import datetime
import lmdb
import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_name")
    parser.add_argument("--data_dir", type=str, default="adsorbate-data/all-adsorbates")
    parser.add_argument("--out_dir", type=str, default="adsorbate-data/inputs")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.1)

    return parser.parse_args()


def get_indices(files):
    return [int(re.findall(r'([\d]*).lmdb', file)[0]) for file in files]

def write_lmdbs(target_path, files):
    out_db = lmdb.open(
        target_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    count=0
    for lmdb_file in tqdm.tqdm(files):
        in_db = lmdb.open(
            lmdb_file,
            readonly=True,
            subdir=False
        )
        with out_db.begin(write=True) as write_txn:
            with in_db.begin() as read_txn:
                cursor = read_txn.cursor()
                for _,value in cursor:
                    write_txn.put(f"{count}".encode("ascii"), value)
                    count += 1
        in_db.close()
        out_db.sync()
    out_db.close()


if __name__ == '__main__':
    args = parse_args()

    ads_files = glob.glob(os.path.join(args.data_dir, "*.lmdb"))
    N = len(ads_files)
    N_val = int(args.val_frac * N)
    N_test = int(args.test_frac * N)
    N_train = N-N_val-N_test

    random.shuffle(ads_files)
    train_files = ads_files[:N_train]
    val_files = ads_files[N_train:N_train + N_val]
    test_files = ads_files[N_train+N_val:]

    outdir = os.path.join(args.out_dir, args.input_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    write_lmdbs(os.path.join(outdir, "train.lmdb"), train_files)
    write_lmdbs(os.path.join(outdir, "val.lmdb"), val_files)
    write_lmdbs(os.path.join(outdir, "test.lmdb"), test_files)

    train_indices, val_indices, test_indices = get_indices(train_files), get_indices(val_files), get_indices(test_files)
    with open(os.path.join(outdir, "indices.txt"), 'w') as outfile:
        outfile.write("train: " + ",".join(train_indices) + "\n")
        outfile.write("val: " + ",".join(val_indices) + "\n")
        outfile.write("test: " + ",".join(test_indices) + "\n")





    