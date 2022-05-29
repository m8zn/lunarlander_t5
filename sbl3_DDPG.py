import gym

from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy

import time
import os

ALGORITHM = "DDPG"
timestamp = int(time.time())

models_dir = f"models/DDPG-{timestamp}"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


# Create environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Instantiate the agent
model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
# Train the agent
TIMESTEPS = 10000
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DDPG-{timestamp}")
    model.save(f"{models_dir}/{TIMESTEPS*i}")       # save model every 10000 steps

env.close()