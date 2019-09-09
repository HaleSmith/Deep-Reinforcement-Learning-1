import gym
from gym import wrappers
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

class FeatureTransformer:
    def __init__(self, n_components, env):
        #Init the scaler
        self.scaler = StandardScaler()
        self.featurizer = FeatureUnion([
                    ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
                    ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
                    ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
                    ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
                    ])

        #Genrerate a bunch of random samples from observation space of the env to
        #fit the scaler and featurizer.
        observation_examples = np.array([env.observation_space.sample() for x in range(5)])
        self.scaler.fit(observation_examples)
        example_features = self.featurizer.fit_transform(observation_examples)

        self.dimensions = example_features.shape[1]

    def transform(self, observation):
            observation = self.scaler.transform(observation)
            return self.featurizer.transform(observation)


class Model:
    def __init__(self, feature_transformer, learning_rate, env):
        self.ft = feature_transformer
        self.models = []
        self.env = env
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])
            self.models.append(model)

    def predict(self, observation):
        feature_vector = self.ft.transform(observation)
        results = []
        for model in self.models:
            results.append(model.predict(feature_vector))
        return results

    def update(self, observation, action, G):
        feature_vector = self.ft.transform(observation)
        self.models[int(action)].partial_fit(feature_vector, [G])

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
        a = model.sample_action([observation], eps)
        prev_obser = observation
        observation,reward,done ,_ = env.step(a)
        total_reward += reward

        #calculating q_sa_max for the next state
        next = model.predict([observation])
        q_sa_max = np.max(next)
        G = reward + gamma*q_sa_max
        model.update([prev_obser], a, G)

    return total_reward

def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    # both X and Y will be of shape (num_tiles, num_tiles)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict([_])), 2, np.dstack([X, Y]))
    # Z will also be of shape (num_tiles, num_tiles)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z,
    rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.show()


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(500, env)
    model = Model(ft, 'constant', env)

    num_of_episodes = 300
    totalrewards = np.empty(num_of_episodes)
    gamma = 0.99

    for n in range(num_of_episodes):
        print ("epi no:", n)
        eps = eps = 0.1*(0.97**n)
        total_reward = play_one_epi(model, eps, gamma)
        totalrewards[n] = total_reward

        if (n + 1) % 100 == 0:
            print("episode:", n, "total reward:", total_reward)
            print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
            print("total steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    # # plot the optimal state-value function
    # plot_cost_to_go(env, model)
    #
    # #play with the trained model to see how it is performing
    # env = gym.make('MountainCar-v0')
    # env = wrappers.Monitor(env, 'mountaincar_vids')
    # model.env = env
    # total_reward = play_one_epi(model, eps, gamma)
