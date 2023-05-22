import pandas as pd
import os
from os.path import join as opj
from os.path import exists as ope
from tqdm import tqdm
from collections import defaultdict



src = 'submission_5fold'
test_data_path = 'test_data.csv'
tar = 'submission_ens'
if not ope(tar):
    os.makedirs(tar)

df = pd.read_csv(test_data_path).rename(columns={'img_path':'uuid'})
for filename in os.listdir(src):
    if 'densenet121' not in filename or 'densenet161' not in filename:
        continue
    tmp = pd.read_csv(opj(src, filename))
    df = pd.concat([df, tmp[['label']].rename(columns={'label':filename})], axis=1)

df['label'] = df.mode(axis=1)[0]
df[['uuid', 'label']].to_csv(opj(tar, f'submission_ens_des121_161.csv'), index=False)