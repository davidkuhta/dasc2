#!/usr/bin/env python
#
# Portions of this code inspired by:
# https://github.com/cole-maclean/autocraft/tree/master/autocraft/EDA/Build%20Orders
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
import ast

from mpyq import MPQArchive
from pkg_resources import resource_filename

def build_order(replay_data, build_orders_dir):
   unit_ids = build_unit_dict()

   '''Script to parse out the build orders from a single state_data file.
   Build orders are the unique units seen from player1's persepective for both players
   in the order they are seen in the replay.
   '''
   build_order = []
   friendly_build = []
   enemy_build = []
   #iterate over state data and find unique units from friendly and enemy player through the states
   for state in replay_data:
       state_build = []
       #load all friendly units seen in this state
       for unit_data in state['friendly_army']:
           #check if unit count > 0
           if unit_data[1] > 0:
               unit = unit_ids[unit_data[0]]
               #lookup unit_id name and append to build if we haven't
               #seen this unit in previous states
               if unit not in friendly_build:
                   friendly_build.append(unit)
                   #append player marker (friendly = 0) to identify player's unit
                   #in build order
                   state_build = state_build + [unit + str(0)]
       for unit_data in state['enemy_army']:
           if unit_data[1] > 0:
               unit = unit_ids[unit_data[0]]
               if unit not in enemy_build:
                   enemy_build.append(unit)
                   state_build = state_build + [unit + str(1)]
       #update build order if new unit seen this state for either player
       if state_build:
           build_order = build_order + state_build
   #gather static data from  first state
   player_id = replay_data[0]['player_id']
   winner = replay_data[0]['winner']
   if player_id == winner:
       won = True
   else:
       won = False
   race = replay_data[0]['race']
   enemy_race = replay_data[0]['enemy_race']
   game_map = replay_data[0]['map_name']
   replay_id = replay_data[0]['replay_id']
   build_data = { "build_order" : build_order, "won" : won, "race" : race,
                  "enemy_race" : enemy_race, "game_map" : game_map,
                  "replay_id" : replay_id }

   json_filename = os.path.join(build_orders_dir, "Build_Orders_" + replay_id[0:20] + "_" + str(player_id) + '.json')
   with open(json_filename, 'w') as json_file:
       json.dump(build_data, json_file)

def format_states_data(states_data):
    text_file = open(states_data,"r")
    lines=text_file.readlines()
    replay_processed = []
    for line in lines:
        replay = ast.literal_eval(line)
        replay_processed.append(replay)
    print(len(replay_processed))
    return replay_processed

def build_unit_dict():
    with open(resource_filename(__name__, '../ref/units.json')) as units_data:
        units = json.load(units_data)
        unit_ids = { int(unit_id) : name for unit_id, name in units.items()}
    return unit_ids
#     units_json = json.loads(units_file)
#     units_dict = {int()}

def build_orders(states_dir, build_orders_dir='./build_orders'):
    # Create directory for replays if it does not exist
    if not os.path.exists(build_orders_dir):
        os.makedirs(build_orders_dir)

    if os.path.exists(states_dir):
        state_files = []

        for file in os.listdir(states_dir):
            if file.endswith(".json"):
                state_files.append(os.path.join(states_dir, file))

        for states_file in state_files:
            formatted_states_file = format_states_data(states_file)
            build_order(formatted_states_file, build_orders_dir)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--states_dir', dest='s_dir', action='store',
                        default='./states', help='Directory where states are stored',
                        required=False)
    parser.add_argument('--build_orders_dir', dest='bo_dir', action='store',
                        default='./build_orders', help='Directory where build orders are to be generated',
                        required=False)


    return parser.parse_args()

def main():
    args = parse_args()
    build_orders(args.s_dir, args.bo_dir)

if __name__ == '__main__':
    main()
