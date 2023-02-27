import gym
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import random

# BUILDING AND ENVIRONMENT

# TODO: Build an agent to give us the best shower possible
# TODO: Randomly temperature
# TODO: We want between 37 - 39 degrees!


class ShowerEnv(gym.Env):
    def __init__(self):
        super(ShowerEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = Discrete(3)  # between 0-2
        self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=np.float32)  # defines initial temperature
        # self.observation_space = Box(low=0, high=100, shape=(1,))  # defines initial temperature
        self.state = 38 + random.randint(-3, 3)  # current temp
        self.shower_length = 60  # 60 second

    def step(self, action):
        # Apply temp adjustment
        # 3 different actions: 0,1,2
        # 0: decreasing the temperature by 1 degree, 1 represents no change, 2 represents increase by 1
        # We already defined the actions space between 0-2
        self.state +=action-1

        # Decrease shower time (at every action the time is decreasing)
        self.shower_length -=1

        # Calculate reward
        if self.state >=37 and self.state <=39:
            reward = 1
        else:
            reward = -1

        # Check whether shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # https://stackoverflow.com/questions/73195438/openai-gyms-env-step-what-are-the-values
        # terminated=True if environment terminates (eg. due to task completion, failure etc.)
        # truncated=True if episode truncates due to a time limit or a reason that is not defined as part of the task MDP.
        info = {}
        truncated = False

        return self.state, reward, done, truncated, info

    def reset(self):
        # reset the shower time to 60 second and current temperature
        self.state = np.array([38 + random.randint(-3,3)],  dtype=np.float32)
        self.shower_length = 60
        return (self.state, {})

    def render(self):
        # Implement visualization
        pass
