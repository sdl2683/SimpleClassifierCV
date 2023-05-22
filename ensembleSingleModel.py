import pandas as pd
import os
from os.path import join as opj
from os.path import exists as ope
from tqdm import tqdm
from collections import defaultdict


def getModelInfo(sub_filename):
    tmp = sub_filename.split('.')[0].split('_')
    tmp.pop(0)
    
    return tmp[0] if 'acc' in tmp[1] else '_'.join(tmp[:2]), tmp[-1]



def submissionClassifier(src):
    subs = os.listdir(src)
    
    sub_infos = defaultdict(list)
    for sub in subs:
        name, fold = getModelInfo(sub)
        sub_infos[name].append(sub)

    return sub_infos


def ensembleSingleModelKFold(src, tar, test_data_path):
    sub_infos = submissionClassifier(src)

    for name, filenames in tqdm(sub_infos.items()):
        df = pd.read_csv(test_data_path).rename(columns={'img_path':'uuid'})
        for filename in filenames:
            tmp = pd.read_csv(opj(src, filename))
            df = pd.concat([df, tmp[['label']].rename(columns={'label':filename})], axis=1)
        
        df['label'] = df.mode(axis=1)[0]

        df[['uuid', 'label']].to_csv(opj(tar, f'submission_{name}_5fold.csv'), index=False)




if __name__ == '__main__':
    src = 'submission_6'
    tar = 'submission_5fold'
    if not ope('submission_5fold'):
        os.makedirs('submission_5fold')

    test_data_path = 'test_data.csv'

    ensembleSingleModelKFold(src, tar, test_data_path)

