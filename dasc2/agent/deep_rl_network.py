# The MIT License (MIT)

# Copyright (c) 2016 Arthur Juliani

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

import tensorflow as tf
import tensorflow.contrib.slim as slim

class DeepRLNetwork():
    def __init__(self, smart_actions, state_size):
        # Deep Qnetwork
        # Based off of exploration strategies (https://github.com/awjuliani/DeepRL-Agents/blob/master/Q-Exploration.ipynb)
        self.inputs = tf.placeholder(shape=[None,state_size],dtype=tf.float32)
        tf.summary.histogram('Rewards', self.inputs)
        self.Temp = tf.placeholder(shape=None,dtype=tf.float32)
        self.keep_per = tf.placeholder(shape=None,dtype=tf.float32)

        hidden = slim.fully_connected(self.inputs,32,activation_fn=None,biases_initializer=None)
        hidden = slim.dropout(hidden,self.keep_per)
        # hidden = tf.nn.rnn_cell.LSTMCell(hidden)
        hidden = slim.fully_connected(hidden,64,activation_fn=tf.nn.tanh,biases_initializer=None)
        hidden = slim.dropout(hidden,self.keep_per)
        hidden = slim.fully_connected(hidden,32,activation_fn=None,biases_initializer=None)
        hidden = slim.dropout(hidden,self.keep_per)
        self.Q_out = slim.fully_connected(hidden,len(smart_actions),activation_fn=None,biases_initializer=None)

        self.predict = tf.argmax(self.Q_out,1)
        self.Q_dist = tf.nn.softmax(self.Q_out/self.Temp)

        tf.summary.histogram('qOut', self.Q_out)
        tf.summary.histogram('qPredict', self.predict)
        # tf.summary.scalar('qDist', self.Q_dist)

        # Calculate loss via squared sum reduction
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,len(smart_actions),dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), reduction_indices=1)

        self.next = tf.placeholder(shape=[None],dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.next - self.Q))
        tf.summary.scalar('loss', loss)
        self.trainer = tf.train.AdagradOptimizer(learning_rate=0.0005)
        self.updateModel = self.trainer.minimize(loss)
        self.merged = tf.summary.merge_all()