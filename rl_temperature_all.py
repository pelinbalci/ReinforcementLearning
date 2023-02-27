# Ref: https://www.youtube.com/watch?v=Mut_u40Sqz4

# INSTALL DEPENDENCIES

# pip install gym
# pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
# pip install tensorboard

import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

#-----------------------------------------------------------------------------------------------------------------------
# TYPES OF SPACES

dis_ex = Discrete(3).sample()
print('discrete ex:2', dis_ex)  # discrete ex:2 1

box_ex = Box(0,1, shape=(3,3)).sample()
print('box ex:', box_ex)

# box ex: [[0.87602866 0.26084232 0.8380754 ]
#  [0.70925903 0.8445923  0.9456141 ]
#  [0.96861917 0.6401565  0.9643551 ]]

tuple_ex = Tuple((Discrete(3), Discrete(3))).sample()
print('tuple ex: ', tuple_ex) # tuple ex:  (2, 1)

dict_ex = Dict({'height': Discrete(2), 'shape': Box(0,100, shape=(1,))})
print('dict_ex', dict_ex)  # dict_ex Dict('height': Discrete(2), 'shape': Box(0.0, 100.0, (1,), float32))

multibin_ex = MultiBinary(4).sample()
print('multibin_ex: ', multibin_ex)  # multibin_ex:  [0 0 1 1]

multidis_ex = MultiDiscrete([5,2,2]).sample()  # between 0-4, 0-1, 0-1
print('multidis_ex: ', multidis_ex)  # multidis_ex:  [4 1 0]


#-----------------------------------------------------------------------------------------------------------------------
# BUILDING AND ENVIRONMENT

# TODO: Buid an agent to give us the best shower possible
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

#-----------------------------------------------------------------------------------------------------------------------
# TEST ENVIRONMENT
env = ShowerEnv()
print('observation space:', env.observation_space.sample())
# observation space: [7.7744584] --> initial temperature

print('action space:', env.action_space.sample())
# action space: 0 --> random action

from stable_baselines3.common.env_checker import check_env
check_env(env, warn=True)

episodes =5
for ep in range(1, episodes+1):
    obs = env.reset()
    done= False
    score = 0

    while not done:
        env.render()  # empty
        action = env.action_space.sample() # choose an action
        obs, reward, done, _, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(ep, score))

env.close()

# Output:
# Episode:1 Score:-10
# Episode:2 Score:-22
# Episode:3 Score:-14
# Episode:4 Score:-60
# Episode:5 Score:-52

#-----------------------------------------------------------------------------------------------------------------------
# TRAIN MODEL

# create model
env = ShowerEnv()
log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

# Output:
# Using cpu device
# Wrapping the env with a `Monitor` wrapper
# Wrapping the env in a DummyVecEnv.

# train model
model.learn(total_timesteps=40000)

#-----------------------------------------------------------------------------------------------------------------------
# SAVE MODEL
model_path = os.path.join('Training', 'SavedModels', 'ShowerPPO')
model.save(model_path)
del model
model = PPO.load(model_path, env)
meanreward, stdreward = evaluate_policy(model, env, n_eval_episodes=10, render=True)

print('mean reward: ', meanreward)
print('standart deviation of reward:', stdreward)

# ----------------------------------------------------------------------------------------------------------------------

print('end')