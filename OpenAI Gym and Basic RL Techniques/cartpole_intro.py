#This file is to calculate the average number of steps before a cartpole episode
#ends when random actions are performed.
import gym

#set the enviromnent to that of cart-pole
env = gym.make('CartPole-v0')
step = 0

for _ in range(1000):
    done = False
    env.reset()
    while not done:
        observation,reward,done ,_ = env.step(env.action_space.sample())
        step += 1


print (step/1000)
