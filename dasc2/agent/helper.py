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

learning_params = {
    "discount" :          .99, # Discount factor.
    "delta" :             0.001, #T arget network per step % change
    "batch_size" :        32, 
    "startE" :            1, # Starting chance of random action
    "endE" :              0.1, # Final chance of random action
    "annealing_steps" :   20000000, #How many steps of training to reduce startE to endE.
    "pre_train_steps" :   64 #Number of steps used before training updates begin.
}

rewards = {
    "SELF_KILLED_UNIT_REWARD" : 0,
    "USED_MINERALS" : 0.001,
    "KILL_UNIT_REWARD" : 0.2,
    "KILL_BUILDING_REWARD" : 0.5
}

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

def build_actions():
    actions = [
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
                actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

    return actions

def updateTargetGraph(tfVars, delta):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*delta) + ((1-delta)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)