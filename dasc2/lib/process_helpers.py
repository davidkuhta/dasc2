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

"""A module containing helper functions for processing SC2 minimaps"""

import numpy as np
import json

from collections import defaultdict
from dasc2.lib.minimap_processor import SC2MinimapProcessor, sf

def process_minimap(minimap, screen):
  """Output a list corresponding to unit density for a given alignment
    Args:
      minimap (np.array):             Minimap from SC2 feature observation
      screen (np.array):            Screen from SC2 feature observation
    Returns:
      (dict) of minimap_data containing information relating the following
      elements within the given camera view:
        "Creep", "Camera" ,"PlayerID", "PlayerRelative", "Power", and
        "UnitDensity", "UnitMode", "HPShields" for "Commanding" and
        "Opposing" alignments
  """
  mp = SC2MinimapProcessor(minimap, screen)

  minimap_data = {  
            "Camera" : minimap[3,:,:].tolist(),
            "PlayerID" : minimap[4,:,:].tolist(),
            "PlayerRelative" : minimap[5,:,:].tolist(),
            "ZergCreep" : minimap[2,:,:].tolist(),
            "ProtossPower" : mp.power(),
            "Commanding" :
            {
              "UnitDensity" : mp.unit_density(1),
              "UnitMode" : mp.unit_mode(1),
              "HPShields" : mp.hp_and_shields(1)
            },
            "Opposing" :
            {
              "UnitDensity" : mp.unit_density(4),
              "UnitMode" : mp.unit_mode(4),
              "HPShields" : mp.hp_and_shields(4)
            }
          }

  return minimap_data

def army_count(screen, alignment):
  """Output a dict corresponding to the units visible for a given player
    Args:
      screen (np.array):  Screen from SC2 feature observation
      alignment (int):  1 (player controlled) or 4 (opponent controlled)
    Returns:
      (dict) of units as key and total screen pixels of that unit type as values.
  """
  # Initialize an empty list to hold units

  army = []
  # Identify units matching the given alignment using a masking technique
  army_units = np.multiply((screen[sf['player_relative']] == alignment).astype(np.int), screen[sf['unit_type']])
  
  # Initialize an empty int dictionary
  army_dict = defaultdict(int)
  # Create a generator to iterate through a flat list of units of the desired alignment
  # that are non-zero and their corresponding pixel counts on the screen

  gen = (unit for unit in zip(army_units.flatten(), screen[sf['unit_density']].flatten()) if unit[0])
  # Implement the generator to construct the new army dictionary
  
  for unit in gen:
    army_dict[str(unit[0])] += unit[1]
    
  return army_dict

class NumpyEncoder(json.JSONEncoder):
  """Modifies JSON encoder to correctly process JSON holding int64"""
  def default(self, obj):
      if isinstance(obj, np.integer):
          return int(obj)
      elif isinstance(obj, np.floating):
          return float(obj)
      elif isinstance(obj, np.ndarray):
          return obj.tolist()
      return json.JSONEncoder.default(self, obj)