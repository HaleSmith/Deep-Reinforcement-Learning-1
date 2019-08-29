import gym
import os
import sys
import numpy as np
from gym import wrappers
import matplotlib.pyplot as plt
from datetime import datetime


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

class FeatureTransformer:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)

    def transform(self, observation):
        cart_pos, cart_vel, pole_angle, pole_vel = observation

        return build_state([to_bin(cart_pos, self.cart_position_bins),
                            to_bin(cart_vel, self.cart_velocity_bins),
                            to_bin(pole_angle, self.pole_angle_bins),
                            to_bin(pole_vel, self.pole_velocity_bins)])

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer
        self.alpha = 0.01

        #Since there are 10 bins, number of states could be their permutations
        num_states = 10**env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.Q = np.random.uniform(low=-1, high=1, size = (num_states, num_actions))

    def predict(self, observation):
        state = self.feature_transformer.transform(observation)
        return self.Q[state]

    def update(self, observation, action, G):
        state = self.feature_transformer.transform(observation)
        self.Q[state,action] += self.alpha*(G - self.Q[state,action])

    def sample_action(self, observation, eps):
        if np.random.uniform() < eps:
            return env.action_space.sample()
        else:
            return np.argmax(self.predict(observation))

def play_one_epi(model, eps, gamma):
    observation = env.reset()
    done = False
    total_reward = 0
    itr = 0

    while not done:
        a = model.sample_action(observation, eps)
        prev_obser = observation
        observation,reward,done ,_ = env.step(a)

        total_reward += reward

        if done and itr < 199:
            reward = -300
        #calculating q_sa_max for the next state
        q_sa_max = np.max(model.predict(observation))
        G = reward + gamma*q_sa_max
        model.update(prev_obser, a, G)
        itr +=  1

    return total_reward

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
      running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.9

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 10000
    totalrewards = np.empty(N)
    for n in range(N):
      eps = 1.0/np.sqrt(n+1)
      totalreward = play_one_epi(model, eps, gamma)
      totalrewards[n] = totalreward
      if n % 100 == 0:
        print("episode:", n, "total reward:", totalreward, "eps:", eps)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)
