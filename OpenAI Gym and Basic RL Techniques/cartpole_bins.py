import gym
import numpy as np
import pandas as pd
from gym import wrappers
import matplotlib.pyplot as plt


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
        cart_pos, cat_vel, pole_angle, pole_vel = observation

        return build_state([to_bin(cart_pos, self.cart_position_bins),
                            to_bin(cart_vel, self.cart_velocity_bins),
                            to_bin(pole_angle, self.cart_angle_bins),
                            to_bin(pole_vel, self.pole_velocity_bins)])
