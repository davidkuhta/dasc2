#!/usr/bin/env python

import os

from dasc2.lib.replay_filter import SC2ReplayFilter


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--min_mmr', dest='mmr', action='store', default=1000, type=int, help='Minimum Player MMR', required=False)
    parser.add_argument('--min_apm', dest='apm', action='store', default=10, type=int, help='Minimum Player APM', required=False)
    parser.add_argument('--races', dest='race', action='store', default=['Zerg','Terr','Prot'], nargs='+',\
        help="Defaults to ['Zerg','Terr','Prot']", required=False)
    parser.add_argument('--game_map', dest='map', action='store', default=None, help='Select a map or default to all', required=False)
    parser.add_argument('--build', dest='vers', action='store', default=None, help='Select a game build version', required=False)
    parser.add_argument('--replay_dir', dest='r_dir', action='store', default=os.getcwd(), help='Replay Directory',required=False)
    
    return parser.parse_args()

def main():
    args = parse_args()
    rf = SC2ReplayFilter(args.mmr, args.apm, args.race, args.map, args.vers)
    rf.filter_replays(args.r_dir)

if __name__ == '__main__':
    main()