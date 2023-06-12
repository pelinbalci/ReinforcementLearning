# Reinforcement Learning Example

My notes from the lecture:  Reinforcement Learning in 3 Hours | Full Course using Python given by 
Nicholas Renotte. You may find the link here: https://www.youtube.com/watch?v=Mut_u40Sqz4

Original Source Code: https://github.com/nicknochnack/ReinforcementLearningCourse/blob/main/Project%203%20-%20Custom%20Environment.ipynb

I highly recommend this great Youtube lecture who would like to understand Refinforcement learning. 

Outline: 

- rl_temperature_all.py : All codes from the lecture (Project -3)
- rl_temperature_env.py: Includes only environment class
- rl_temperature_train.py: Train and save the model
- rl_temperature_livetest.py: Predict and observe the output


Install Dependencies:
 - pip install gym
- pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
- pip install tensorboard


In this repository I used the codes for the Project- 3 in the lecture and make some changes do to the API change.
Some of the changes:

1- step function returns 5 values instead of 4. There is a new returned value after new API which is "truncated"
2- I got many errors because of the shape differences between env.reset() and observation_space. I've added dtype 
to both of the values to solve this problem: 

    self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=np.float32) 
    self.state = np.array([38 + random.randint(-3,3)],  dtype=np.float32)

3- During the environment check, I got this error: 

    AssertionError: `reset()` must return a tuple (obs, info)

In order to handle ths error, I simply add an empty dictionary to the return value of reset function. Please reach me to
handle this error on a proper way. (I added this problem on issues)

        def reset(self):
            # reset the shower time to 60 second and current temperature
            self.state = np.array([38 + random.randint(-3,3)],  dtype=np.float32)
            self.shower_length = 60
            return (self.state, {})


