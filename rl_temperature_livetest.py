import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from rl_temp_class import ShowerEnv

env = ShowerEnv()
model_path = os.path.join('Training', 'SavedModels', 'ShowerPPO')
model = PPO.load(model_path, env)
meanreward, stdreward = evaluate_policy(model, env, n_eval_episodes=10, render=True)

print('mean reward: ', meanreward)
print('standart deviation of reward:', stdreward)

episodes =2
for ep in range(1, episodes+1):
    print('episode:', ep)
    obs = env.reset()
    obs = obs[0]
    done= False
    reward_sum = 0

    while not done:
        current_state = env.state[0]
        action, _state = model.predict(obs)

        #  0: decreasing the temperature by 1 degree, 1 represents no change, 2 represents increase by 1
        if action == 1:
            action_str = 'no_change'
        elif action == 2:
            action_str = 'increase'
        else:
            action_str = 'decrease'
        obs, reward, done, _, info = env.step(action)
        reward_sum +=reward
        print('state:',  current_state, 'action:', action, '-', action_str, 'obs:', obs, 'cum_reward:', reward_sum)