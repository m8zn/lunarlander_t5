import gym

from stable_baselines3 import DQN, DDPG
from stable_baselines3.common.evaluation import evaluate_policy

import datetime
import os

import imageio
import numpy as np

env = gym.make('LunarLanderContinuous-v2')
env.reset()

### define model directory
models_dir = "models/DDPG-1653853310"

### define step 
model_path = f"{models_dir}/980000.zip"

# model = DQN.load(model_path, env=env)
model = DDPG.load(model_path, env=env)

rewards_ep = []

images = []
img = model.env.render(mode='rgb_array')

episodes = 10 

for ep in range(episodes):
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        images.append(img)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        img = model.env.render(mode='rgb_array')
        rewards.append(reward)
    
    print(f"episode: {ep} reward sum: {np.array(rewards).sum()}")
    rewards_ep.append(np.array(rewards).sum())

print(f"reward mean: {np.array(rewards_ep).mean()}")

imageio.mimsave('lander_ddpg-1653853310.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
env.close()
