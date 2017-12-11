#Built on top of https://github.com/skjb/pysc2-tutorial/blob/master/Building%20an%20Attack%20Agent/attack_agent.py

import random
import math

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
import tensorflow as tf
import tensorflow.contrib.slim as slim

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

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT1 = 'buildsupplydepot1'
ACTION_BUILD_SUPPLY_DEPOT2 = 'buildsupplydepot2'
ACTION_BUILD_BARRACKS1 = 'buildbarracks1'
ACTION_BUILD_BARRACKS2 = 'buildbarracks2'
ACTION_SELECT_BARRACKS1 = 'selectbarracks1'
ACTION_SELECT_BARRACKS2 = 'selectbarracks1'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'
ACTION_BUILD_SCV = 'buildscv'
ACTION_SELECT_CC = 'selectcc'
ACTION_SCV_MINERALS = 'scvminerals'
ACTION_SCV_VESPENE = 'scvvespene'

# smart_actions = [
#     ACTION_DO_NOTHING,
#     ACTION_SELECT_SCV,
#     ACTION_BUILD_SUPPLY_DEPOT,
#     ACTION_BUILD_BARRACKS1,
#     ACTION_SELECT_BARRACKS1,
#     ACTION_BUILD_MARINE,
#     ACTION_SELECT_ARMY,
#     ACTION_BUILD_SCV,
#     ACTION_SELECT_CC,
#     ACTION_SCV_MINERALS,
#     ACTION_SCV_VESPENE,
# ]

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT1,
    ACTION_BUILD_SUPPLY_DEPOT2,
    ACTION_BUILD_BARRACKS1,
    ACTION_BUILD_BARRACKS2,
    ACTION_SELECT_BARRACKS1,
    ACTION_SELECT_BARRACKS2,
    ACTION_BUILD_MARINE,
    ACTION_SELECT_ARMY,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

# out["score_cumulative"] = np.array([
#     obs.score.score,
#     obs.score.score_details.idle_production_time,
#     obs.score.score_details.idle_worker_time,
#     obs.score.score_details.total_value_units,
#     obs.score.score_details.total_value_structures,
#     obs.score.score_details.killed_value_units,
#     obs.score.score_details.killed_value_structures,
#     obs.score.score_details.collected_minerals,
#     obs.score.score_details.collected_vespene,
#     obs.score.score_details.collection_rate_minerals,
#     obs.score.score_details.collection_rate_vespene,
#     obs.score.score_details.spent_minerals,
#     obs.score.score_details.spent_vespene,
# ], dtype=np.int32)


KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5
# COLLECTION_RATE_MINERALS = out["score_cumulative"][9]/100
# COLLECTION_RATE_VESPENE = out["score_cumulative"][10]/100
# USED_MINERALS = out["score_cumulative"][11]/100 # out["score_cumulative"][11]
# USED_VESPENE = out["score_cumulative"][12]/100 # out["score_cumulative"][12]
# SELF_KILLED_UNIT_REWARD = -0.1 # Some negative reward, out["score_cumulative"][3], I think
# SELF_KILLED_BUILDING_REWARD = -0.3 # Some negative reward, out["score_cumulative"][4], I think

# Set learning parameters
exploration = "boltzmann" #Exploration method. Choose between: greedy, random, e-greedy, boltzmann, bayesian.
discount = .99 #Discount factor.
num_episodes = 20000 #Total number of episodes to train network for.
tau = 0.001 #Amount to update target network at each step.
batch_size = 32 #Size of training batch
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 200000 #How many steps of training to reduce startE to endE.
pre_train_steps = 1500 #Number of steps used before training updates begin.

class experience_buffer():
    def __init__(self, buffer_size = 10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,4])

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

class deepRLNetwork():
    def __init__(self, smart_actions, state_size):
        # Deep Qnetwork
        # Based off of exploration strategies (https://github.com/awjuliani/DeepRL-Agents/blob/master/Q-Exploration.ipynb)
        self.inputs = tf.placeholder(shape=[None,state_size],dtype=tf.float32)
        self.Temp = tf.placeholder(shape=None,dtype=tf.float32)
        self.keep_per = tf.placeholder(shape=None,dtype=tf.float32)

        hidden = slim.fully_connected(self.inputs,64,activation_fn=tf.nn.tanh,biases_initializer=None)
        hidden = slim.dropout(hidden,self.keep_per)
        self.Q_out = slim.fully_connected(hidden,len(smart_actions),activation_fn=tf.nn.sigmoid,biases_initializer=None)

        self.predict = tf.argmax(self.Q_out,1)
        self.Q_dist = tf.nn.softmax(self.Q_out/self.Temp)


        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,len(smart_actions),dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), reduction_indices=1)

        self.nextQ = tf.placeholder(shape=[None],dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Q))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
        self.updateModel = trainer.minimize(loss)

    def choose_action(self, state):
        pass
    def learn(self, s, a, r ,s_):
        pass
# class Qnetwork():
#     def __init__(self, ):

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class AttackAgent(base_agent.BaseAgent):
    def __init__(self):
        super(AttackAgent, self).__init__()


        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_used_minerals = 0

        self.previous_action = None
        self.previous_state = None

        self.state_size = 20

        tf.reset_default_graph()
        self.qlearn = deepRLNetwork(smart_actions, self.state_size)
        self.target_net = deepRLNetwork(smart_actions, self.state_size)

        self.init = tf.global_variables_initializer()
        self.trainables = tf.trainable_variables()
        self.targetOps = updateTargetGraph(self.trainables,tau)
        self.myBuffer = experience_buffer()

        self.sess = tf.Session()
        self.sess.run(self.init)
        updateTarget(self.targetOps,self.sess)
        self.e = startE
        self.stepDrop = (startE-endE)/annealing_steps
        self.total_steps = 0

        # self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

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
        supply_depot_count = supply_depot_count = 1 if depot_y.any() else 0

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = 1 if barracks_y.any() else 0

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

        hot_squares = np.zeros(16)
        enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 16):
            current_state[i + 4] = hot_squares[i]

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD

            if total_used_minerals > self.previous_used_minerals:
                reward += (total_used_minerals - self.previous_used_minerals)/10

            self.myBuffer.add(np.reshape(np.array([self.previous_state,self.previous_action,reward,current_state]),[1,4]))

            if self.e > endE and self.total_steps > pre_train_steps:
                self.e -= self.stepDrop

            if self.total_steps > pre_train_steps and self.total_steps % 5 == 0:
                #We use Double-DQN training algorithm
                trainBatch = self.myBuffer.sample(batch_size)
                Q1 = self.sess.run(self.qlearn.predict,feed_dict={self.qlearn.inputs:np.vstack(trainBatch[:,3]),self.qlearn.keep_per:5.0})
                Q2 = self.sess.run(self.target_net.Q_out,feed_dict={self.target_net.inputs:np.vstack(trainBatch[:,3]),self.target_net.keep_per:5.0})
                doubleQ = Q2[range(batch_size),Q1]
                targetQ = trainBatch[:,2] + (discount*doubleQ)
                _ = self.sess.run(self.qlearn.updateModel,feed_dict={self.qlearn.inputs:np.vstack(trainBatch[:,0]),self.qlearn.nextQ:targetQ,self.qlearn.keep_per:1.0,self.qlearn.actions:trainBatch[:,1]})
                updateTarget(self.targetOps,self.sess)
            # self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        # rl_action = self.qlearn.choose_action(str(current_state))
        # rl_action = self.sess.run([self.qlearn.predict, self.qlearn.Q_out], feed_dict = {self.qlearn.inputs:current_state, self.qlearn.Temp:e})

        if exploration == "greedy":
            #Choose an action with the maximum expected value.
            a,allQ = self.sess.run([self.qlearn.predict,self.qlearn.Q_out],feed_dict={self.qlearn.inputs:[current_state],self.qlearn.keep_per:1.0})
            rl_action = a[0]
        if exploration == "random":
            #Choose an action randomly.
            rl_action = np.random.choice(len(smart_actions))
        if exploration == "e-greedy":
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < self.e or self.total_steps < pre_train_steps:
                rl_action = np.random.choice(len(smart_actions))
            else:
                a,allQ = self.sess.run([self.qlearn.predict,self.qlearn.Q_out],feed_dict={self.qlearn.inputs:[current_state],self.qlearn.keep_per:1.0})
                rl_action = a[0]
        if exploration == "boltzmann":
            #Choose an action probabilistically, with weights relative to the Q-values.
            Q_d,allQ = self.sess.run([self.qlearn.Q_dist,self.qlearn.Q_out],feed_dict={self.qlearn.inputs:[current_state],self.qlearn.Temp:self.e,self.qlearn.keep_per:1.0})
            a = np.random.choice(Q_d[0],p=Q_d[0])
            rl_action = np.argmax(Q_d[0] == a)
        if exploration == "bayesian":
            #Choose an action using a sample from a dropout approximation of a bayesian q-network.
            a,allQ = self.sess.run([self.qlearn.predict,self.qlearn.Q_out],feed_dict={self.qlearn.inputs:[current_state],self.qlearn.keep_per:(1-self.e)+0.1})
            rl_action = a[0]

        smart_action = smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_used_minerals = total_used_minerals
        self.previous_state = current_state
        self.previous_action = rl_action
        self.total_steps += 1

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        if smart_action == ACTION_DO_NOTHING:
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

        elif smart_action == ACTION_SELECT_BARRACKS1:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                loc = np.random.choice([0,20])
                target = self.transformDistance(int(unit_x.mean()), 20, int(unit_y.mean()), loc)

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

        elif smart_action == ACTION_SELECT_BARRACKS2:
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
