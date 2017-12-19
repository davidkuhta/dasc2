#!/usr/bin/env python
#
# Copyright 2017 David Kuhta & Anshul Sacheti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time

from mpyq import MPQArchive

from dasc2.lib.replay_filter import SC2ReplayFilter
from dasc2.lib.replay_helpers import check_build_version

def filter_replays(replay_filter):

    replays_dir = os.path.abspath(replay_filter.replays_dir)

    total_replay_counter = 0
    filtered_replay_counter = 0

    replay_list = []
    replay_metadata_list = []

    file_list = os.listdir(replays_dir)

    try:
        from tqdm import tqdm
        file_list = tqdm(file_list)
    except ImportError:
        pass

    for file in file_list:

        if file.endswith(".SC2Replay"):
            total_replay_counter += 1
            replay = os.path.join(replays_dir, file)
            replay_metadata = replay_filter.filter(replay)

            if replay_metadata:
                filtered_replay_counter += 1
                filename = os.path.splitext(file)[0]
                replay_metadata_list.append({ filename : replay_metadata })

    if total_replay_counter:
        print(str(filtered_replay_counter) + " of " + str(total_replay_counter) + " replays met filtering criteria\n")
    else:
        print("No SC2 Replays to evaluate in this directory")

    return replay_metadata_list

def generate_filter_file(replay_filter, replay_list):
    
    filter_dir = os.path.abspath(replay_filter.filter_dir)

    if not os.path.exists(filter_dir):
        os.makedirs(filter_dir)

    json_filename = os.path.join(filter_dir, replay_filter.name + "_filter" + ".json")

    with open(json_filename, 'w') as json_file:

        json_output = {}

        json_output.update({ 'Name' : replay_filter.name})
        json_output.update({ 'CreatedAt' : time.strftime("%Y%m%d-%H%M%S")})
        json_output.update({ 'DataBuild' : replay_filter.game_version})

        if replay_filter.map_title:
            map_title = replay_filter.map_title
        else:
            map_title = 'all'

        json_output.update({ 'Maps' : map_title})


        json_output.update({ 'MinMMR' : replay_filter.mmr_threshold })
        json_output.update({ 'MinAPM' : replay_filter.apm_threshold })
        json_output.update({ 'Races' : replay_filter.races })
        json_output.update({ 'WinningRaces' : replay_filter.winning_races })
        json_output.update({ 'Replays' : replay_list })
        json_output.update({ 'ReplaysDirectory' : os.path.abspath(replay_filter.replays_dir) })

        json.dump(json_output, json_file)

        print("Replay Filter generated")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', dest='name', action='store', default='SC2Filter',
                        help='Filter Name', required=False)
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

    processed_version = int(check_build_version(args.vers, False))

    replay_filter = SC2ReplayFilter(args.mmr, args.apm, args.races, args.winning_races,
                        args.map, processed_version, args.r_dir, args.f_dir, args.name)

    replay_filter._info()

    filtered_replay_list = filter_replays(replay_filter)
    if filtered_replay_list:
        generate_filter_file(replay_filter, filtered_replay_list)

if __name__ == '__main__':
    main()
