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

"""Generate unit/technology build orders from replay states"""

import os
import json
import argparse
import collections

from pkg_resources import resource_filename

def __create_unit_dict():
    """Constructs a dictionary of unit_ids to unit_labels
        
        Returns:
            Dictionary: { unit_id : unit_name } <int> : <string>
    """

    with open(resource_filename(__name__, '../data/units_simple.json')) as units_data:
        units = json.load(units_data)
        unit_ids = { int(unit_id) : name for unit_id, name in units.items()}
    return unit_ids

def __units_seen(states, army_type):
    """Identifies units seen, adding to dictionary in the order seen

        Args:
            states (list):          list of game states.
            army_type (string):     "Commanding" or "Opposing".
        
        Returns:
            Dictionary: { unit_id : step_when_seen } <int> : <int>
    """
    units_seen_dict = collections.OrderedDict()
    for state in states:
        for unit in state['Armies'][army_type]:
            if not unit in units_seen_dict.keys():
                units_seen_dict[unit] = state['Step']

    return units_seen_dict

def __label_units(units_dict, unit_id_dict):
    """Remaps the keys of dictionary to their label in another dictionary

        Args:
            units_dict (dict):      Dictionary of unit_ids to steps_when_seen.
            unit_id_dict (dict):    Dictionary of unit_ids to unit_labels.
        
        Returns:
            Dictionary: { unit_id : step_when_seen } <int> : <int>

        Example:
        With a dictionary of {... 43 : "Viking" ... }
        we would remap a units_dict of:
            { ... 43 : 800 ... } to { ... "Viking" : 800 ... }
    """
    labeled_units_seen_dict = collections.OrderedDict()
    for unit_id, step in units_dict.items():
        label = unit_id_dict[int(unit_id)]
        labeled_units_seen_dict[label] = step

    return labeled_units_seen_dict

def build_order(states, unit_id_dict):
    """Generates the build data for one replay

        Args:
            states (list):          List of states for a particular replay.
            unit_id_dict (dict):    Dictionary to use for relabeling.
        
        Returns:
            Dictionary: { "Commanding" : ..., "Opposing" : ... } <string> : <list>
    """
    commanding_b_o = __units_seen(states, 'Commanding')
    opposing_b_o = __units_seen(states, 'Opposing')
    labeled_commanding_b_o = __label_units(commanding_b_o, unit_id_dict)
    labeled_opposing_b_o = __label_units(opposing_b_o, unit_id_dict)

    build_data = { "Commanding " : list(labeled_commanding_b_o.keys()),
                    "Opposing" : list(labeled_opposing_b_o.keys()) }

    return build_data

def build_orders(states_dir='./states', build_orders_dir='./build_orders'):
    """Process as state files in a directory, generating build orders for each

        Args:
            states_dir (str):       Directory containing a list of states.
            build_orders_dir (str): Directory to output build orders.
    """

    # If states directory exists
    if os.path.exists(states_dir):

        # Create directory for replays if it does not exist
        if not os.path.exists(build_orders_dir):
            os.makedirs(build_orders_dir)

        state_files = []
        unit_id_dict = __create_unit_dict()

        # Iterate through files in states directory evaluating
        # only those that are json and appending to a running list
        for file in os.listdir(states_dir):
            if file.endswith(".json"):
                state_files.append(os.path.join(states_dir, file))

        # We'll try to use tqdm for pretty printing a status bar
        # which will succeed if the user selected to install it
        # during the `pip install dasc2`
        try:
            from tqdm import tqdm
            state_files = tqdm(state_files)
        except ImportError:
            pass

        # Iterate through the list of state files
        for states_file in state_files:

            # load the replay state data as a JSON object
            replay_state_data = json.load(open(states_file))

            # pass the ['States'] element to the build_order function
            build_data = build_order(replay_state_data['States'], unit_id_dict)

            # Duplicate the JSON state data object
            bo_data = replay_state_data

            # Remove the ['States'] element as we've already processed the data
            del bo_data['States']

            # Inject a new ['ArmyBuildOrder'] element using the processed build data
            bo_data['ArmyBuildOrder'] = build_data

            # Construct a new JSON file
            states_filename = os.path.basename(states_file)
            bo_filename = os.path.join(build_orders_dir, "BO_" + states_filename)

            # Write out the JSON file
            with open(bo_filename, 'w') as json_file:
                json.dump(bo_data, json_file)
                json_file.write('\n')

def parse_args():
    """Helper function to parse dasc2_build_order arguments"""

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