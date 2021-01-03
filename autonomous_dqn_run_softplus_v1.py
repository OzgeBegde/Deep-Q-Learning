# -*- coding: utf-8 -*-
"""autonomous_dqn_run_v1_relu.py

"""

from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
import numpy as np
from octasonic import Octasonic
import sys
from time import time
import os
from collections import namedtuple
from math import fabs

from DQN_softplus import DQN_softplus
from DQN_softplus import load_table
from DQN_softplus import save_table
##from DQN_softplus import calculate_reward

from actions_dqn import servo_install
from actions_dqn import get_state
from actions_dqn import action_an
from actions_dqn import recovery

# Front and Back wheel classes instantiated
bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()
picar.setup()

# Front wheel Setup
fw.offset = 0
fw.turn(86.86)

fw_angle = 86.86

octasonic = Octasonic(0)
octasonic.set_sensor_count(8)

crash_dist = 12
epsilon = 0.7   # The randomness of chosen action, used for discovery.

corner_sensor_margin = 5    # Maybe dimming the corner sensors makes movements close to walls more possible.

crash_counter = 0
crash_list = []

dqn = DQN_softplus(5,15,3)

sensor_num=5
hidden_layer_size=15
action_num=3
    
##w1 = np.random.rand(sensor_num, hidden_layer_size)
##w2 = np.random.rand(hidden_layer_size, action_num)



def main(dqn):
    global crash_list
    epochs = 0
    
    print "DQN_softplus Values set."
    
    decay_epsilon = epsilon
    weights_list = [] 
    start_time = time()
    
    crash_list.append(str(0)+","+str(start_time)+"\n")
        
    while True:
        print "\n-----\nEpoch: ", epochs + 1, "\n-----"
        epoch_start = time()
##        crash_list.append(str(epochs + 1)+","+ str(time()))
        crash_report = run(dqn, decay_epsilon, start_time)
        bw.stop()
        crash_list.append(str(epochs + 1)+","+ str(epoch_start)+","+ str(time())+","+str(time()-epoch_start)+"\n")
        print(type(crash_report))
##        if len(crash_report)== 0:
##            last_actions_list = []
##        else:
        last_actions_list = [sa_t.action for sa_t in crash_report]
        sleep(1)
        recovery(last_actions_list, fw,  bw, octasonic)
        epochs += 1
        sleep(1)
        if decay_epsilon > 0.005:
            decay_epsilon -= 0.005   # decaying epsilon
        print "\n\nvDEBUG Epsilon:", decay_epsilon
        
            


# The schedule of a single epoch. This is the core program.
def run(dqn, decay_epsilon, start_time):    
    global crash_counter
    global crash_list
    
    crash = False
       # Fresh buffer for every epoch

    #sa_tuple = namedtuple('sa_tuple', 'state action next_state reward')
    total_reward=0

    state_action_buffer = []    # Fresh buffer for every epoch

    sa_tuple = namedtuple('sa_tuple', 'state action next_state reward')
    while not crash:
               
        print "---\nCycle\n---"

        current_input = get_state(octasonic)
        print "vDEBUG Raw Sensor Input:", current_input
        # Immediate sensor checks to stop the car in crash situations. 
        if is_crash(current_input[:3]):
            print("\nCrash!\n")
            bw.stop()
            break
        
        current_state = convert_input_to_state(current_input)      
        print "vDEBUG Current state: ", current_state


        if np.random.rand() > decay_epsilon:
            print "\nBEST action chosen."
            next_action = dqn.get_best_action(current_state)
        else:
            print "\nRANDOM action chosen."
            next_action = dqn.get_random_action()
            
        print "Action chosen: ", next_action

        action_an(next_action, fw, bw)
# 3 <
        next_input = get_state(octasonic)

        # Immediate sensor checks to stop the car in crash situations. 
        if is_crash(next_input[:3]):
            print("---\nCrash!\n---")
            bw.stop()
            crash = True
        
        next_state = convert_input_to_state(next_input)
        next_reward = dqn.calculate_reward(next_state)
        new_sa_tuple = sa_tuple(state=current_state, action=next_action, reward=next_reward, next_state=next_state)
        state_action_buffer.append(new_sa_tuple)
              
        total_reward=total_reward+next_reward

##        q_est=dqn.computeQEstimate(next_state, next_reward)
##        print q_est
##        w_1=q_est[2] 
##        w_2=q_est[3]
        weights=dqn.Grad_des(next_state)
##        print ('state action buffer',state_action_buffer)
        
##        weights_list.append(weights)
    return state_action_buffer[-5:]

        
    

# Kalacak
# Calculates the reward value that will be used in the q-function.
# Actually, this method of the program directly dictates the policies that the agent is going to inherit.


# Kalacak
# Converts the sensor input that's between 0-255, to limited state-space that's between 0-2
def convert_input_to_state(input):
    state_list = []
    for i in input:
        if i <= crash_dist:
            state_list.append(-1)
        elif crash_dist <= i and i <= 20:
            state_list.append(0)
        elif 21 <= i and i <= 32:
            state_list.append(1)
        elif i > 32:
            state_list.append(2)
    return state_list
####################################
# Kalacak
def is_crash(state):
    if state[0] <= crash_dist - corner_sensor_margin:
        print "Crash on LEFT corner."
        return True
    elif state[1] <= crash_dist:
        print "Crash up FRONT."
        return True
    elif state[2] <= crash_dist - corner_sensor_margin:
        print "Crash on RIGHT corner."
        return True
    return False

# Kalacak
def update_crash_log(crash_list):
    string = "[\n" + str(crash_list[0])
    
    for item in crash_list[1:]:
        string = string + str(item)
    string = string + "]\n"
        
    f=open("crash_log_dqn.txt", "a+")
    f.write(string)
    f.close()


# Kalacak
if __name__ == "__main__":
    save_weights = raw_input("Do you want to load from save? (Y/N)\n")
    try:
##        dqn = DQN_softplus(5,15,3)
        if save_weights.upper() == 'Y':
            w1_pre,w2_pre = load_table()
            dqn = DQN_softplus(5,15,3,w1_pre,w2_pre)
            
        main(dqn)
    except KeyboardInterrupt:
        bw.stop()
        save = raw_input("Do you want to save the last session's weihts? (Y/N)\n")
        if save.upper() == 'Y':
            save_table(dqn)
            print "Table saved.\n\n"
        print "This session's crash counting:", str(crash_list)
        save_c = raw_input("\nDo you want to store the crash list in the Crash Log? (Y/N)\n")
        if save_c.upper() == 'Y':
            update_crash_log(crash_list)
        print "\nHave a nice day.\n"
        




