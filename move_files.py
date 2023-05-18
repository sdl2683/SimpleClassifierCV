import shutil, os
from tqdm import tqdm
from os.path import join as opj
import argparse


def move_files(src, tar, keywords):
    
    files = os.listdir(src)

    for each in tqdm(files):
        if any([each.find(keyword)!=-1 for keyword in keywords]):
            shutil.move(opj(src, each), opj(tar, each))
            print(f'move {each} to {tar} successfully!')

def copy_files(src, tar, keywords):
    
    files = os.listdir(src)

    for each in tqdm(files):
        if any([each.find(keyword)!=-1 for keyword in keywords]):
            shutil.copy(opj(src, each), opj(tar, each))
            print(f'copy {each} to {tar} successfully!')
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', help='source path of the files need to move.')
    parser.add_argument('--tar', help='target path of the files need to move.')
    parser.add_argument('--keywords', help='keywords of the files need to move.')
    parser.add_argument('--mode', default='move', choices=['move', 'copy'], help='mode of operation, move or copy.')

    args = parser.parse_args()


    if args.mode == 'move':
        move_files(args.src, args.tar, args.keywords)
    elif args.mode == 'copy':
        copy_files(args.src, args.tar, args.keywords)

if __name__ == '__main__':
    main()