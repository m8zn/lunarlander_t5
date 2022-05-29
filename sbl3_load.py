import gym

from stable_baselines3 import DQN, DDPG
from stable_baselines3.common.evaluation import evaluate_policy

import datetime
import os

env = gym.make('LunarLanderContinuous-v2')
env.reset()

### define model directory
models_dir = "models/DDPG-1653841974"

### define step 
model_path = f"{models_dir}/570000.zip"

# model = DQN.load(model_path, env=env)
model = DDPG.load(model_path, env=env)


episodes = 10 

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render()

env.close()
