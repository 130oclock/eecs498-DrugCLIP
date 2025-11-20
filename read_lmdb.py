import os
import argparse

import pickle
import lmdb
import selfies as sf
from tqdm import tqdm, trange

def read_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )  
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    out_list = []
    for idx in tqdm(keys):
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        out_list.append(data)
        break

    env.close()
    return out_list


with open('dude-pocket.txt', 'w') as f:
    out_list = read_lmdb("./data/DUD-E/raw/all/aa2ar/pocket.lmdb")
    for data in out_list:
        for key in data:
            f.write(repr(key) + ":" + str(data[key]) + '\n')

with open('ours-pocket.txt', 'w') as f:
    out_list = read_lmdb("./data/custom/aa2ar/pocket.lmdb")
    for data in out_list:
        for key in data:
            f.write(repr(key) + ":" + str(data[key]) + '\n')

with open('dude-mols.txt', 'w') as f:
    out_list = read_lmdb("./data/DUD-E/raw/all/aa2ar/mols.lmdb")
    for data in out_list:
        for key in data:
            f.write(repr(key) + ":" + str(data[key]) + '\n')

with open('ours-mols.txt', 'w') as f:
    out_list = read_lmdb("./data/custom/aa2ar/mols.lmdb")
    for data in out_list:
        for key in data:
            f.write(repr(key) + ":" + str(data[key]) + '\n')