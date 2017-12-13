#!/usr/bin/env python

import os
import argparse
import subprocess

import sys
sys.path.insert(0, '../')

def extract_replays(archives_dir, replays_dir, password, remove=False):
    # Create directory for replays if it does not exist
    if not os.path.exists(replays_dir):
        os.makedirs(replays_dir)

    if os.path.exists(archives_dir):
        archives = []

        for file in os.listdir(archives_dir):
            if file.endswith(".zip"):
                archives.append(os.path.join(archives_dir, file))

        for archive in archives:
            s = '7z e {0} -o{1} -p{2}'.format(archive, replays_dir, password)
            subprocess.call([s], shell=True)
            if remove:
                os.remove(archive)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archives_dir', dest='archives_dir', action='store', type=str, default='./archives',
                        help='Directory containing replay archives')
    parser.add_argument('--replays_dir', dest='replays_dir', action='store', type=str, default='./replays',
                        help='Directory in which to extract replays')
    parser.add_argument('--password', dest='password', action='store', type=str, default='./replays',
                        help='password for the archives')
    parser.add_argument('--remove_archives', dest='remove', action='store_true',
                        help='remove archives after extraction')
    return parser.parse_args()

def main():
    args = parse_args()
    extract_replays(args.archives_dir, args.replays_dir, args.password, args.remove)


if __name__ == '__main__':
    main()
