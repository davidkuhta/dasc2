#!/usr/bin/python
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

"""Dump out stats about all the actions that are in use in a set of replays."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import multiprocessing
import os
import signal
import sys
import threading
import time

from future.builtins import range 
import six
from six.moves import queue

def sorted_dict_str(d):
  return "{%s}" % ", ".join("%s: %s" % (k, d[k])
                            for k in sorted(d, key=d.get, reverse=True))

class ReplayStats(object):
  """Summary stats of the replays seen so far."""

  def __init__(self):
    self.replays = 0
    self.steps = 0
    self.camera_move = 0
    self.select_pt = 0
    self.select_rect = 0
    self.control_group = 0
    self.maps = collections.defaultdict(int)
    self.races = collections.defaultdict(int)
    self.unit_ids = collections.defaultdict(int)
    self.valid_abilities = collections.defaultdict(int)
    self.made_abilities = collections.defaultdict(int)
    self.valid_actions = collections.defaultdict(int)
    self.made_actions = collections.defaultdict(int)
    self.crashing_replays = set()
    self.invalid_replays = set()

  def merge(self, other):
    """Merge another ReplayStats into this one."""
    def merge_dict(a, b):
      for k, v in six.iteritems(b):
        a[k] += v

    self.replays += other.replays
    self.steps += other.steps
    self.camera_move += other.camera_move
    self.select_pt += other.select_pt
    self.select_rect += other.select_rect
    self.control_group += other.control_group
    merge_dict(self.maps, other.maps)
    merge_dict(self.races, other.races)
    merge_dict(self.unit_ids, other.unit_ids)
    merge_dict(self.valid_abilities, other.valid_abilities)
    merge_dict(self.made_abilities, other.made_abilities)
    merge_dict(self.valid_actions, other.valid_actions)
    merge_dict(self.made_actions, other.made_actions)
    self.crashing_replays |= other.crashing_replays
    self.invalid_replays |= other.invalid_replays

  def __str__(self):
    len_sorted_dict = lambda s: (len(s), sorted_dict_str(s))
    len_sorted_list = lambda s: (len(s), sorted(s))
    return "\n\n".join((
        "Replays: %s, Steps total: %s" % (self.replays, self.steps),
        "Camera move: %s, Select pt: %s, Select rect: %s, Control group: %s" % (
            self.camera_move, self.select_pt, self.select_rect,
            self.control_group),
        "Maps: %s\n%s" % len_sorted_dict(self.maps),
        "Races: %s\n%s" % len_sorted_dict(self.races),
        "Unit ids: %s\n%s" % len_sorted_dict(self.unit_ids),
        "Valid abilities: %s\n%s" % len_sorted_dict(self.valid_abilities),
        "Made abilities: %s\n%s" % len_sorted_dict(self.made_abilities),
        "Valid actions: %s\n%s" % len_sorted_dict(self.valid_actions),
        "Made actions: %s\n%s" % len_sorted_dict(self.made_actions),
        "Crashing replays: %s\n%s" % len_sorted_list(self.crashing_replays),
        "Invalid replays: %s\n%s" % len_sorted_list(self.invalid_replays),
    ))


class ProcessStats(object):
  """Stats for a worker process."""

  def __init__(self, proc_id, step_mul=8):
    self.proc_id = proc_id
    self.time = time.time()
    self.stage = ""
    self.replay = ""
    self.replay_stats = ReplayStats()
    self.step_mul = step_mul

  def update(self, stage):
    self.time = time.time()
    self.stage = stage

  def __str__(self):
    return ("[%2d] replay: %10s, replays: %5d, steps: %7d, game loops: %7s, "
            "last: %12s, %3d s ago" % (
                self.proc_id, self.replay, self.replay_stats.replays,
                self.replay_stats.steps,
                self.replay_stats.steps * self.step_mul, self.stage,
                time.time() - self.time))

def stats_printer(stats_queue, parallel, flag_print_time):
  """A thread that consumes stats_queue and prints them every 10 seconds."""
  proc_stats = [ProcessStats(i) for i in range(parallel)]
  print_time = start_time = time.time()
  width = 107

  running = True
  while running:
    print_time += flag_print_time #10

    while time.time() < print_time:
      try:
        s = stats_queue.get(True, print_time - time.time())
        if s is None:  # Signal to print and exit NOW!
          running = False
          break
        proc_stats[s.proc_id] = s
      except queue.Empty:
        pass

    replay_stats = ReplayStats()
    for s in proc_stats:
      replay_stats.merge(s.replay_stats)

    print((" Summary %0d secs " % (print_time - start_time)).center(width, "="))
    print(replay_stats)
    print(" Process stats ".center(width, "-"))
    print("\n".join(str(s) for s in proc_stats))
    print("=" * width)