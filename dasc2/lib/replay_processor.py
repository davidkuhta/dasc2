#!/usr/bin/python
#
# portions referenced from Deepmind/pysc2 replay_actions.py
# in accordance with the license below:
#
# Copyright 2017 Google Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import multiprocessing
import os
import signal
import sys
import threading
import time #? needed

from future.builtins import range
from websocket import _exceptions
import six
from six.moves import queue

from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import protocol
from pysc2.lib import remote_controller

from absl import app
from absl import flags
from pysc2.lib import gfile
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb

from dasc2.lib.stats import ProcessStats
from dasc2.lib.process_helpers import army_count, process_minimap, NumpyEncoder

import numpy as np
import json

screen_size = point.Point(84, 84)
minimap_size = point.Point(16, 16)
interface = sc_pb.InterfaceOptions(
    raw=True, score=False,
    feature_layer=sc_pb.SpatialCameraSetup(width=24))
screen_size.assign_to(interface.feature_layer.resolution)
minimap_size.assign_to(interface.feature_layer.minimap_resolution)

result_dict = { "Victory" : True, "Defeat" : False }

def valid_replay(info, ping, replay, replay_list):
    """Make sure the replay isn't corrupt, and is worth looking at."""
    if info.HasField("error"):
        print((" Replay has error field ").center(107, "="))
        return False
    if (info.base_build != ping.base_build or  # different game version
        info.game_duration_loops < 1000 or
        len(info.player_info) != 2):
    # Probably corrupt, or just not interesting.
        return False
    if replay[0:20] in replay_list:
        print((" Replay has been processed ").center(107, "="))
        return False
    return True

class ReplayProcessor(multiprocessing.Process):
  """A Process that pulls replays and processes them."""

  def __init__(self, proc_id, run_config, replay_queue,
               stats_queue, step_mul, states_dir='./states', winner_only=False):
    super(ReplayProcessor, self).__init__()
    self.step_mul = step_mul
    self.stats = ProcessStats(proc_id, self.step_mul)
    self.run_config = run_config
    self.replay_queue = replay_queue
    self.stats_queue = stats_queue
    self.states_dir = states_dir
    self.winner_only = winner_only
    self.replay_list = self.saved_replay_list()

  def saved_replay_list(self):
    """Formulate a list of all previously processed replay files"""
    replays = []
    for root, dirs, files in os.walk(self.states_dir):
        for name in files:
            replays.append(name[0:20])
    return replays

  def run(self):
    signal.signal(signal.SIGTERM, lambda a, b: sys.exit())  # Exit quietly.
    self._update_stage("spawn")
    replay_name = "none"
    while True:
      self._print("Starting up a new SC2 instance.")
      self._update_stage("launch")
      try:
        with self.run_config.start() as controller:
          self._print("SC2 Started successfully.")
          ping = controller.ping()
          for _ in range(300):
            try:
              replay_path = self.replay_queue.get()
              print(replay_path)
            except queue.Empty:
              self._update_stage("done")
              self._print("Empty queue, returning")
              return
            try:
              replay_name = os.path.basename(replay_path)#[:10]
              self.stats.replay = replay_name
              self._print("Got replay: %s" % replay_path)
              self._update_stage("open replay file")
              replay_data = self.run_config.replay_data(replay_path)
              self._update_stage("replay_info")
              try:
                  info = controller.replay_info(replay_data)
              except _exceptions.WebSocketTimeoutException:
                  self._print("Replay timedout in run.")
                  break
              self._print((" Replay Info %s " % replay_name).center(60, "-"))
              self._print(info)
              self._print("-" * 60)
              if valid_replay(info, ping, replay_name, self.replay_list):
                self.stats.replay_stats.maps[info.map_name] += 1

                map_data = None
                if info.local_map_path:
                  self._update_stage("open map file")
                  map_data = self.run_config.map_data(info.local_map_path)

                for player_id in [0, 1]:
                  race = sc_common.Race.Name(info.player_info[player_id].player_info.race_actual)
                  self.stats.replay_stats.races[race] += 1
                  enemy_race = sc_common.Race.Name(info.player_info[int(not(player_id))].player_info.race_actual)
                  won = info.player_info[player_id].player_result.result
                  if won == 1 or not(self.winner_only):
                    try:
                        self._print("Starting %s from player %s's perspective" % (
                            replay_name, player_id))
                        self.process_replay(controller, replay_data, map_data,
                                            player_id, replay_name, info.map_name,
                                            won, race, enemy_race)
                    except _exceptions.WebSocketTimeoutException:
                        self._print("Replay timedout in run.")
                        pass
              else:
                self._print("Replay is invalid.")
                #self.stats.replay_stats.invalid_replays.add(replay_name)
            finally:
              self.replay_queue.task_done()
          self._update_stage("shutdown")
      except (protocol.ConnectionError, protocol.ProtocolError,
              remote_controller.RequestError):
        pass
        #self.stats.replay_stats.crashing_replays.add(replay_name)
      except KeyboardInterrupt:
        return

  def _print(self, s):
    for line in str(s).strip().splitlines():
      print("[%s] %s" % (self.stats.proc_id, line))

  def _update_stage(self, stage):
    self.stats.update(stage)
    self.stats_queue.put(self.stats)

  def process_replay(self, controller, replay_data, map_data,
                     player_id, replay_id, map_name, won,
                     race, enemy_race):
    """Process a single replay, updating the stats."""
    self._update_stage("start_replay")
    try:
        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id + 1))
    except _exceptions.WebSocketTimeoutException:
        self._print("Replay timedout in process_replay.")
        return

    #clear datafile
    states_file = os.path.join(self.states_dir, replay_id[0:20] + "_" + str(player_id) + '.json')
    with open(states_file, 'w') as outfile: pass

    feat = features.Features(controller.game_info())

    self.stats.replay_stats.replays += 1
    self._update_stage("step")
    controller.step()
    
    # Initialize dict for states
    state_list = []
    step = 0
    while True:
      step += 1
      self.stats.replay_stats.steps += 1

      self._update_stage("observe")
      obs = controller.observe()
      actions = []
      for action in obs.actions:
        act_fl = action.action_feature_layer
        if act_fl.HasField("unit_command"):
          self.stats.replay_stats.made_abilities[
              act_fl.unit_command.ability_id] += 1
        if act_fl.HasField("camera_move"):
          self.stats.replay_stats.camera_move += 1
        if act_fl.HasField("unit_selection_point"):
          self.stats.replay_stats.select_pt += 1
        if act_fl.HasField("unit_selection_rect"):
          self.stats.replay_stats.select_rect += 1
        if action.action_ui.HasField("control_group"):
          self.stats.replay_stats.control_group += 1

        try:
          full_act = feat.reverse_action(action)
          func = feat.reverse_action(action).function
          args = full_act.arguments
        except ValueError:
          func = -1
          args = []


        self.stats.replay_stats.made_actions[func] += 1
        actions.append([func, args])

      for valid in obs.observation.abilities:
        self.stats.replay_stats.valid_abilities[valid.ability_id] += 1

      for u in obs.observation.raw_data.units:
        self.stats.replay_stats.unit_ids[u.unit_type] += 1

      for ability_id in feat.available_actions(obs.observation):
        self.stats.replay_stats.valid_actions[ability_id] += 1

      all_features = feat.transform_obs(obs.observation)

      # #remove elevation, viz and selected data from minimap

      # minimap_data = all_features['minimap'][2:6,:,:]
      # screen = all_features['screen']

      # mini_shape = minimap_data.shape
      # minimap = np.zeros(shape=(11,mini_shape[1],mini_shape[2]),dtype=np.int)
      # minimap[0:4,:,:] = minimap_data
      # extended_minimap = update_minimap(minimap,screen).tolist()
        
      # Retrieve army counts

      minimap = all_features['minimap']
      screen = all_features['screen']

      minimap_data = process_minimap(minimap, screen)

      commanding_army = army_count(screen, 1)
      opposing_army = army_count(screen, 4)

      full_state = { "Step": step, "MinimapData": minimap_data,
                    "Armies" : { "Commanding" : commanding_army, "Opposing" : opposing_army } ,
                    "AllFeatPlayer":all_features['player'].tolist(), 
                    "AllFeatAvailActions":all_features['available_actions'].tolist(), "Actions":actions }
    
      # Append state to list
      state_list.append(full_state)

      if step % 100 == 0:
        print(" Just added step %s " % step)

      if obs.player_result:
        print("Generating final file")
        # Generate the final state and output to json file
        final_state = { "ReplayID" : replay_id, "MapName" : map_name,
                        "PlayerID" : player_id, "Won": won, "Race" : race,
                        "EnemyRace" : enemy_race, "States" : state_list }
        with open(states_file, 'a') as outfile:
            json.dump(final_state, outfile, cls=NumpyEncoder)

        break

      self._update_stage("step")
      controller.step(self.step_mul)