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

import numpy as np
from collections import defaultdict

screen_features = [ "height_map", "visibility_map", "creep", "power", "player_id",
"player_relative", "unit_type", "selected", "unit_hit_points",
"unit_hit_points_ratio", "unit_energy", "unit_energy_ratio", "unit_shields",
"unit_shields_ratio", "unit_density", "unit_density_aa", "effects" ]

minimap_features = [ "height_map", "visibility_map", "creep", "camera", "player_id",
"player_relative", "selected"]

sf = {k: v for v, k in enumerate(screen_features)}
mf = {k: v for v, k in enumerate(minimap_features)}

def army_count(screen, player):
  army = []
  army_units = np.multiply((screen[sf['player_relative']] == player).astype(np.int), screen[sf['unit_type']])
  # counter = np.count_nonzero(army_units)
  # if counter:
  #   print("Counter %s for player %s" % (counter, player))
  army_dict = defaultdict(int)
  gen = (unit for unit in zip(army_units.flatten(), screen[sf['unit_density']].flatten()) if unit[0])
  for unit in gen:
    army_dict[str(unit[0])] += unit[1]
  return army_dict

def update_minimap(minimap, screen):
    #Update minimap data with screen details
    #Identify which minimap squares are on screen

    visible = minimap[1] == 1

    #TODO: need to devide screen into visible minimap, for now
    #divide each quantity by number of visible minimap squares
    total_visible = sum(visible.ravel())

    #power
    minimap[4,visible] = (sum(screen[sf['power']].ravel())/
                          (len(screen[sf['power']].ravel())*total_visible))

    #friendy army
    friendly_units = screen[sf['player_relative']] == 1

    #unit density
    minimap[5,visible] = sum(screen[sf['unit_density'],friendly_units])/total_visible
    #Most common unit
    if friendly_units.any() == True:
        minimap[6,visible] = np.bincount(screen[sf['unit_type'],friendly_units]).argmax()
    else:
        minimap[6,visible] = 0
    #Total HP + Shields
    minimap[7,visible] = ((sum(screen[sf['unit_hit_points'],friendly_units]) +
                          sum(screen[sf['unit_shields'],friendly_units]))/total_visible)
    #enemy army
    enemy_units = screen[sf['player_relative']] == 4
    #unit density
    minimap[8,visible] = sum(screen[sf['unit_density'],enemy_units])/total_visible
    #main unit
    if enemy_units.any() == True:
        minimap[9,visible] = np.bincount(screen[sf['unit_type'],enemy_units]).argmax()
    else:
        minimap[9,visible] = 0
    #Total HP + shields
    minimap[10,visible] = ((sum(screen[sf['unit_hit_points'],enemy_units]) +
                            sum(screen[sf['unit_shields'],friendly_units]))/total_visible)

    return minimap