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

"""SC2Replay minimap processing object module"""

import numpy as np

screen_features = [ "height_map", "visibility_map", "creep", "power", "player_id",
					"player_relative", "unit_type", "selected", "unit_hit_points",
					"unit_hit_points_ratio", "unit_energy", "unit_energy_ratio", "unit_shields",
					"unit_shields_ratio", "unit_density", "unit_density_aa", "effects" ]

minimap_features = [ "height_map", "visibility_map", "creep", "camera", "player_id",
					 "player_relative", "selected"]
processed_minimap_features = [ "creep", "camera", "player_id", "player_relative",
							  "power", "com_unit_density", "com_unit_mode", "com_hp_shields",
							  "opp_unit_density", "opp_unit_mode", "com_hp_shields" ]

sf = {k: v for v, k in enumerate(screen_features)}
mf = {k: v for v, k in enumerate(minimap_features)}
pmf = {k: v for v, k in enumerate(processed_minimap_features)}

class SC2MinimapProcessor(object):
	"""A Process that manipulates SC2 Minimap data."""

	def __init__(self, minimap, screen):
		"""Initialize a SC2MinimapProcessor
			Args:
				minimap (np.array):	Minimap from SC2 feature observation
				screen (np.array):	Screen from SC2 feature observation
		"""
		self.camera = minimap[mf['camera']] == 1
		self.camera_sum = sum(self.camera.ravel())
		self.mm_shape = minimap.shape
		self.screen = screen
		
	def power(self):
		"""Output a list corresponding to protoss power
			Returns:
				(list) minimap view of protoss power
		"""
		power_minimap = np.zeros(shape=(self.mm_shape[1], self.mm_shape[2]), dtype=np.int)

		power_sum = sum(self.screen[sf['power']].ravel())
		averaged_sum = len(self.screen[sf['power']].ravel()) * self.camera_sum

		if averaged_sum:
			end_sum = power_sum / averaged_sum
		else:
			end_sum = 0
		power_minimap[self.camera] = power_sum / averaged_sum

		return power_minimap.tolist()

	def unit_density(self, alignment):
		"""Output a list corresponding to unit density for a given alignment
			Args:
				alignment (int):	1 (player controlled) or 4 (opponent controlled)
			Returns:
				(list) minimap view of unit density for given alignment
		"""
		ud_minimap = np.zeros(shape=(self.mm_shape[1], self.mm_shape[2]), dtype=np.int)

		aligned_units = self.screen[sf['player_relative']] == alignment
		aligned_density_sum = sum(self.screen[sf['unit_density'], aligned_units])

		ud_minimap[self.camera] = aligned_density_sum  / self.camera_sum

		return ud_minimap.tolist()

	def unit_mode(self, alignment):
		"""Output a list corresponding to highest frequency unit for a given alignment
			Args:
				alignment (int):	1 (player controlled) or 4 (opponent controlled)
			Returns:
				(list) minimap view of unit mode for given alignment
		"""
		um_minimap = np.zeros(shape=(self.mm_shape[1], self.mm_shape[2]), dtype=np.int)

		aligned_units = self.screen[sf['player_relative']] == alignment
		
		if aligned_units.any():
			unit_mode = np.bincount(self.screen[sf['unit_type'], aligned_units]).argmax()
			um_minimap[self.camera] = unit_mode
		
		return um_minimap.tolist()

	def hp_and_shields(self, alignment):
		"""Output a list corresponding to average hp and shields for a given alignment
			Args:
				alignment (int):	1 (player controlled) or 4 (opponent controlled)
			Returns:
				(list) minimap view of hp and shields for given alignment
		"""
		hs_minimap = np.zeros(shape=(self.mm_shape[1], self.mm_shape[2]), dtype=np.int)
		
		aligned_units = self.screen[sf['player_relative']] == alignment

		hp_sum = sum(self.screen[sf['unit_hit_points'], aligned_units])
		shields_sum = sum(self.screen[sf['unit_shields'], aligned_units])
		hp_and_shields_sum = hp_sum + shields_sum
		
		hs_minimap[self.camera] = hp_and_shields_sum / self.camera_sum

		return hs_minimap.tolist()