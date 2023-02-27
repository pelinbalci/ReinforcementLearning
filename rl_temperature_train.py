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
from rl_temp_class import ShowerEnv

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
