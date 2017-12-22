# portions of code derived from https://github.com/skjb/pysc2-tutorial attack_agent.py
# in accordance with the license below:
#
# MIT License

# Copyright (c) 2017 Steven Brown

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import math
import os

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
import tensorflow as tf
# import tensorflow.contrib.slim as slim
from tensorflow.python.ops import rnn, rnn_cell

from dasc2.agent.exp import exp
from dasc2.agent.helper import build_actions, updateTarget, updateTargetGraph, learning_params as lp
from dasc2.agent.deep_rl_network import DeepRLNetwork

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_NOT_QUEUED = [0]
_QUEUED = [1]

ACTION_NO_OP = 'no-op'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT1 = 'buildsupplydepot1'
ACTION_BUILD_SUPPLY_DEPOT2 = 'buildsupplydepot2'
ACTION_BUILD_BARRACKS1 = 'buildbarracks1'
ACTION_BUILD_BARRACKS2 = 'buildbarracks2'
ACTION_SELECT_BARRACKS1 = 'selectbarracks1'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'
ACTION_BUILD_SCV = 'buildscv'
ACTION_SELECT_CC = 'selectcc'
ACTION_SCV_MINERALS = 'scvminerals'
ACTION_SCV_VESPENE = 'scvvespene'

smart_actions = [
    ACTION_NO_OP,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT1,
    ACTION_BUILD_SUPPLY_DEPOT2,
    ACTION_BUILD_BARRACKS1,
    ACTION_BUILD_BARRACKS2,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_SELECT_ARMY,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

SELF_KILLED_UNIT_REWARD = 0
USED_MINERALS = 0.001
KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5

class AttackAgent(base_agent.BaseAgent):
    def __init__(self):
        super(AttackAgent, self).__init__()

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_used_minerals = 0
        self.previous_army_supply = 0

        self.previous_action = None
        self.previous_state = None

        self.state_size = 22

        # Initializations/Tensorboard
        tf.reset_default_graph()
        self.qlearn = DeepRLNetwork(smart_actions, self.state_size)
        self.target_net = DeepRLNetwork(smart_actions, self.state_size)

        self.init = tf.global_variables_initializer()
        self.trainables = tf.trainable_variables()
        self.targetOps = updateTargetGraph(self.trainables,lp["delta"])
        self.myBuffer = exp(10000)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        # Tensorboard
        # self.merged = tf.summary.merge_all()
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        self.train_writer = tf.summary.FileWriter('./outputs',self.sess.graph)

        self.sess.run(self.init)
        updateTarget(self.targetOps,self.sess)
        self.e = lp["startE"]
        self.stepDrop = (lp["startE"]-lp["endE"])/lp["annealing_steps"]
        self.total_steps = 0

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

    def step(self, obs):
        super(AttackAgent, self).step(obs)

        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        if depot_x.any():
            supply_depot_count = np.ceil((max(depot_x)-min(depot_x))/8.)
        else:
            supply_depot_count = 0
        # supply_depot_count = supply_depot_count = 1 if depot_y.any() else 0

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        if barracks_y.any():
            barracks_count = np.ceil((max(barracks_y)-min(barracks_y))/10.)
        else:
            barracks_count = 0
        # barracks_count = 1 if barracks_y.any() else 0

        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]

        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]

        total_used_minerals = obs.observation['score_cumulative'][11]

        current_state = np.zeros(self.state_size)
        current_state[0] = supply_depot_count
        current_state[1] = barracks_count
        current_state[2] = supply_limit
        current_state[3] = army_supply
        current_state[4] = self.base_top_left
        current_state[5] = self.total_steps

        hot_squares = np.zeros(16)
        enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 16):
            current_state[i + 6] = hot_squares[i]

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD * (killed_unit_score - self.previous_killed_unit_score)

            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD * (killed_building_score - self.previous_killed_building_score)

            if total_used_minerals > self.previous_used_minerals:
                reward += (total_used_minerals - self.previous_used_minerals)/USED_MINERALS

            if self.previous_army_supply > army_supply:
                tmp = SELF_KILLED_UNIT_REWARD * (self.previous_army_supply - army_supply)
                if (reward - tmp) > 0:
                    reward -= tmp
                else:
                    reward = 0

            self.myBuffer.append(np.reshape(np.array([self.previous_state,self.previous_action,reward,current_state]),[1,4]))

            if self.e > lp["endE"] and self.total_steps > lp["pre_train_steps"]:
                self.e -= self.stepDrop

            if self.total_steps > lp["pre_train_steps"] and self.total_steps % 10 == 0:
                #We use Double-DQN training algorithm
                trainBatch = self.myBuffer.query(lp["batch_size"])
                Q_a = self.sess.run(self.qlearn.predict,feed_dict={self.qlearn.inputs:np.vstack(trainBatch[:,3]),self.qlearn.keep_per:5.0})
                Q_b = self.sess.run(self.target_net.Q_out,feed_dict={self.target_net.inputs:np.vstack(trainBatch[:,3]),self.target_net.keep_per:5.0})
                Q_c = Q_b[range(lp["batch_size"]),Q_a]
                Q_t = (lp["discount"]*Q_c) + trainBatch[:,2]
                summary, _ = self.sess.run([self.qlearn.merged, self.qlearn.updateModel],feed_dict={self.qlearn.inputs:np.vstack(trainBatch[:,0]),self.qlearn.next:Q_t,self.qlearn.keep_per:5.0,self.qlearn.actions:trainBatch[:,1]},options=self.run_options,run_metadata=self.run_metadata)
                updateTarget(self.targetOps,self.sess)

                if self.total_steps >= 1000:
                    # print("Saving model")
                    self.train_writer.add_run_metadata(self.run_metadata, 'step%d' % self.total_steps)
                    self.train_writer.add_summary(summary, self.total_steps)
                    if self.total_steps % 1000 == 0:
                        self.saver.save(self.sess, './outputs')
            # self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        # rl_action = self.qlearn.choose_action(str(current_state))
        # rl_action = self.sess.run([self.qlearn.predict, self.qlearn.Q_out], feed_dict = {self.qlearn.inputs:current_state, self.qlearn.Temp:e})

        #bayesian q-network to sample for actions
        a,allQ = self.sess.run([self.qlearn.predict,self.qlearn.Q_out],feed_dict={self.qlearn.inputs:[current_state],self.qlearn.keep_per:(1-self.e)+0.1})
        rl_action = a[0]

        smart_action = smart_actions[rl_action]

        # print(smart_action)
        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_used_minerals = total_used_minerals
        self.previous_army_supply = army_supply
        self.previous_state = current_state
        self.previous_action = rl_action
        self.total_steps += 1

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        if smart_action == ACTION_NO_OP:
            return actions.FunctionCall(_NO_OP, [])

        elif smart_action == ACTION_SELECT_SCV:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT1:
            if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                if unit_y.any():
                    target = self.transformDistance(int(unit_x.mean()), 0, int(unit_y.mean()), 20)

                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT2:
            if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                if unit_y.any():

                    target = self.transformDistance(int(unit_x.mean()), -15, int(unit_y.mean()), 20)

                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

        elif smart_action == ACTION_BUILD_BARRACKS1:
            if _BUILD_BARRACKS in obs.observation['available_actions']:
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                if unit_y.any():
                    target = self.transformDistance(int(unit_x.mean()), 20, int(unit_y.mean()), 20)

                    return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        elif smart_action == ACTION_BUILD_BARRACKS2:
            if _BUILD_BARRACKS in obs.observation['available_actions']:
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                if unit_y.any():
                    target = self.transformDistance(int(unit_x.mean()), 20, int(unit_y.mean()), 0)

                    return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        elif smart_action == ACTION_SELECT_BARRACKS:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                loc = np.random.choice([0,20])
                target = self.transformDistance(int(unit_x.mean()), 20, int(unit_y.mean()), loc)

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

        elif smart_action == ACTION_BUILD_MARINE:
            if _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        elif smart_action == ACTION_ATTACK:
            if obs.observation['single_select'][0][0] != _TERRAN_SCV and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x), int(y))])

        return actions.FunctionCall(_NO_OP, [])