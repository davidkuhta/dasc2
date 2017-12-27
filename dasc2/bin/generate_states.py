#!/usr/bin/python
#
# portions utilized from Deepmind/pysc2 replay_actions.py in accordance
# with the license below:
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
from websocket import _exceptions

from pysc2 import run_configs

from absl import app
from absl import flags
from pysc2.lib import gfile

from dasc2.lib.stats import ReplayStats, stats_printer
from dasc2.lib.replay_processor import ReplayProcessor

import json

# Maintain abseil for consistency with Google's pysc2
FLAGS = flags.FLAGS
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_integer("step_mul", 8, "How many game steps per observation.")
flags.DEFINE_string("states_dir", "./states", "Path to place state data")
flags.DEFINE_integer("print_time", 100, "Interval between stat prints and data saves in seconds")

flags.DEFINE_string("filter_file", None, "Filter File containing replay list.")
flags.mark_flag_as_required("filter_file")
flags.DEFINE_bool("winner_only", False, "Process Replays for both winner and loser")

FLAGS(sys.argv)

def replay_queue_filler(replay_queue, replay_list):
  """A thread that fills the replay_queue with replay filenames."""
  for replay_path in replay_list:
    replay_queue.put(replay_path)

def main():

  """Dump stats about all the actions that are in use in a set of replays."""
  run_config = run_configs.get()

  filter_file = FLAGS.filter_file
  if not gfile.Exists(filter_file):
    sys.exit("{} doesn't exist.".format(filter_file))
    
    print("Generating replay list using:", filter_file)
    
  # Instantiate empty replay list
  replay_list = []
  
  # Import filter JSON
  filtered_data = json.load(open(filter_file))
  
  # Identify replay directory from filter
  replay_dir = filtered_data['ReplaysDirectory']
  
  # Iterate through replays denoted in filter
  for replay in filtered_data['Replays']:
    for key in replay:
      if key:
        # Generate the replay filename
        replay_file = os.path.join(replay_dir, key + '.SC2Replay')
        # Append replay file path to replay list
        replay_list.append(replay_file)  
  
  # Ensure the path for the states directory exists
  # else create it.
  if not os.path.exists(FLAGS.states_dir):
    os.makedirs(FLAGS.states_dir)

  stats_queue = multiprocessing.Queue()
  stats_thread = threading.Thread(target=stats_printer,
                                  args=(stats_queue, FLAGS.parallel, FLAGS.print_time))
  stats_thread.start()
  try:
    # For some reason buffering everything into a JoinableQueue makes the
    # program not exit, so save it into a list then slowly fill it into the
    # queue in a separate thread. Grab the list synchronously so we know there
    # is work in the queue before the SC2 processes actually run, otherwise
    # The replay_queue.join below succeeds without doing any work, and exits.
    
    replay_queue = multiprocessing.JoinableQueue(FLAGS.parallel * 10)
    replay_queue_thread = threading.Thread(target=replay_queue_filler,
                                           args=(replay_queue, replay_list))
    replay_queue_thread.daemon = True
    replay_queue_thread.start()

    for i in range(FLAGS.parallel):
      p = ReplayProcessor(i, run_config, replay_queue, stats_queue,
                          FLAGS.step_mul, FLAGS.states_dir, FLAGS.winner_only)
      p.daemon = True
      p.start()
      time.sleep(1)  # Stagger startups, otherwise they seem to conflict somehow

    replay_queue.join()  # Wait for the queue to empty.
  except KeyboardInterrupt:
    print("Caught KeyboardInterrupt, exiting.")
  finally:
    stats_queue.put(None)  # Tell the stats_thread to print and exit.
    stats_thread.join()

if __name__ == "__main__":
  main()