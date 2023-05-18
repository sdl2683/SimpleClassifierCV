import pandas as pd
import os
from os.path import join as opj
from os.path import exists as ope
from tqdm import tqdm

src = 'submission_6'
subs = os.listdir(src)

df = pd.read_csv('submission_ens_dense169_effb0_effb1_effb2_effb3.csv')[['uuid']]

for sub in tqdm(subs):
    if 'efficientnet_b0' not in sub:
        tmp = pd.read_csv(opj(src, sub))
        df = pd.concat([df, tmp[['label']].rename(columns={'label':sub})], axis=1)
        df['label'] = df.mode(axis=1)[0]

df[['uuid', 'label']].to_csv('submission_des169_b1_b2_b3_rext50.csv',index=False)
