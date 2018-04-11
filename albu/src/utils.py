import threading
import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from sklearn.model_selection import KFold
from config import Config
import argparse
import json
import pandas as pd


def get_folds(data, num):
    kf = KFold(n_splits=num, shuffle=True, random_state=42)
    kf.get_n_splits(data)
    return kf.split(data)


def get_csv_folds(ds, fn, holdout=False):
    df = pd.read_csv(fn)
    # df = df[df['source'] != 'janowczyk']
    df = df[['img_id', 'fold', 'cluster', 'source']]
    folds = []
    polosa_id = '193ffaa5272d5c421ae02130a64d98ad120ec70e4ed97a72cdcd4801ce93b066'
    galaxy_ids = ['538b7673d507014d83af238876e03617396b70fe27f525f8205a4a96900fbb8e', 'a102535b0e88374bea4a1cfd9ee7cb3822ff54f4ab2a9845d428ec22f9ee2288', 'cb4df20a83b2f38b394c67f1d9d4aef29f9794d5345da3576318374ec3a11490', 'f29fd9c52e04403cd2c7d43b6fe2479292e53b2f61969d25256d2d2aca7c6a81']
    all_folds_ids = galaxy_ids + [polosa_id]
    for it in set(df['fold']):
        toadd = (df['fold'] != it) | (df['img_id'].isin(all_folds_ids)) | (df['source'] == 'wikimedia')

        val = df[(df['fold'] == it)]['img_id'].values.tolist()
        train_groups = df[toadd]['cluster'].values
        train_ids = df[toadd]['img_id'].values

        train = []
        for i in range(len(train_ids)):
            rep = 1
            if train_groups[i] in ['b', 'd', 'e', 'm']:
                rep = 4
            elif train_groups[i] in ['c', 'n']:
                rep = 3
            if train_ids[i] == polosa_id:
                rep = 5
            train.extend([train_ids[i]] * rep)
        folds.append((ds.get_indexes_by_names(train), ds.get_indexes_by_names(val)))
    return folds


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--fold', dest='fold', default=None,
                        help='fold to process')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
        print(config)
    config['fold'] = args.fold
    return Config(**config)

def update_config(config, **kwargs):
    d = config._asdict()
    d.update(**kwargs)
    print(d)
    return Config(**d)