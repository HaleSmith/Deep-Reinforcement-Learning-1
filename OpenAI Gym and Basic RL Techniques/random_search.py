'''
This implementation aims at solving a cartpole problem using random search
method. For every set of random parameters we observe the total rewards obtained
and select the best set. This is basically randomly searching for the best
hyperparameters in R^n, where n is the dimension of the parameters.
'''

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

def determine_action(params, state):
    action = np.dot(params,state)
    if action > 0:
        return 1
    return 0

w_min = 0
w_max = 1

env = gym.make('CartPole-v0')

max_steps = -10000000
data = []
for i in range(100):
    params = np.random.uniform(w_min,w_max,4)
    done = False
    state = env.reset()
    steps = 0
    while not done:
        action = determine_action(params, state)
        state,reward,done ,_ = env.step(action)
        steps += 1
    data.append(steps)
    if steps > max_steps:
        max_steps = steps
        optimum_param = params

x_axis = range(100)

plt.plot(x_axis, data)
plt.show()
