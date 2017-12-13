#!/usr/bin/env python
#
# Copyright (c) 2017 David Kuhta & Anshul Sacheti
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import json
import os
import time

from mpyq import MPQArchive

from dasc2.lib.replay_filter import SC2ReplayFilter

def filter_replays(replay_filter, replays_dir='./replays',
                    filters_dir='./filters', full_path=False):

    replays_dir = os.path.abspath(replays_dir)
    filters_dir = os.path.abspath(filters_dir)

    if not os.path.exists(filters_dir):
        os.makedirs(filters_dir)

    metadata_dir = filters_dir + "/metadata"
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    replay_filter._info()

    time_string = "_" + time.strftime("%Y%m%d-%H%M%S")
    mmr_string = "_MMR_" + str(replay_filter.mmr_threshold)
    apm_string = "_APM_" + str(replay_filter.apm_threshold)

    filtered_name = "Filtered_Replays"+ mmr_string + apm_string + time_string
    json_filename = os.path.join(metadata_dir, filtered_name + ".json")
    txt_filename =  os.path.join(filters_dir, filtered_name + ".txt")

    total_replay_counter = 0
    filtered_replay_counter = 0

    with open(json_filename, 'w') as json_file, open(txt_filename, 'w') as txt_file:
        replay_metadata_list = []

        file_list = os.listdir(replays_dir)
        PROGRESS_BAR_FLAG = False

        try:
            from tqdm import tqdm
            file_list = tqdm(file_list)
            PROGRESS_BAR_FLAG = True
        except ImportError:
            pass

        for filename in file_list:

            if filename.endswith(".SC2Replay"):
                total_replay_counter += 1
                replay = os.path.join(replays_dir, filename)
                replay_metadata = replay_filter.filter(replay)

                if replay_metadata:
                    filtered_replay_counter += 1
                    if full_path:
                        txt_file.write(replay+"\n")
                    else:
                        txt_file.write(filename+"\n")
                    replay_json = {str(filename.split(".")[0]):replay_metadata}
                    replay_metadata_list.append(replay_json)

        json.dump(replay_metadata_list, json_file)

    if total_replay_counter:
        print(str(filtered_replay_counter) + " of " + str(total_replay_counter) + " replays met filtering criteria\n")
    else:
        print("No SC2 Replays to evaluate in this directory")

    if filtered_replay_counter:
        print("List of replays written to: " + str(txt_filename))
        print("JSON data for replays written to: " + str(json_filename))
    else:
        os.remove(json_filename)
        os.remove(txt_filename)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--min_mmr', dest='mmr', action='store', default=1000,
                        type=int, help='Minimum Player MMR', required=False)
    parser.add_argument('--min_apm', dest='apm', action='store', default=10,
                        type=int, help='Minimum Player APM', required=False)
    parser.add_argument('--races', dest='races', action='store',
                        default=['Zerg','Terr','Prot'], nargs='+',
                        help="Defaults to: Zerg Terr Prot", required=False)
    parser.add_argument('--winning_races', dest='winning_races', action='store',
                        default=['Zerg','Terr','Prot'], nargs='+',
                        help="Defaults to: Zerg Terr Prot", required=False)
    parser.add_argument('--game_map', dest='map', action='store', default=None,
                        help='Select a map or default to all', required=False)
    parser.add_argument('--build', dest='vers', action='store', default=None,
                        help='Select a game build version', required=False)
    parser.add_argument('--replays_dir', dest='r_dir', action='store',
                        default='./replays', help='Directory where replays are stored',
                        required=False)
    parser.add_argument('--filters_dir', dest='f_dir', action='store',
                        default='./filters', help='Directory where replays are stored',
                        required=False)
    parser.add_argument('--full_path', action='store_true',
                        help='Generate full replay path')

    return parser.parse_args()

def main():
    args = parse_args()
    replay_filter = SC2ReplayFilter(args.mmr, args.apm, args.races, args.winning_races,
                        args.map, args.vers)

    filter_replays(replay_filter, args.r_dir, args.f_dir, args.full_path)

if __name__ == '__main__':
    main()
