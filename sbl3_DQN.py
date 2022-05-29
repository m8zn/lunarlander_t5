import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from rewardWrapper import RewardWrapper

import time
import os

ALGORITHM = "DQN"
timestamp = int(time.time())

models_dir = f"models/DQN-{timestamp}"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


# Create environment
# env = RewardWrapper(gym.make('LunarLander-v2'))
env = gym.make('LunarLander-v2')
env.reset()

# Instantiate the agent
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
# Train the agent
TIMESTEPS = 10000
for i in range(1,60):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DQN-{timestamp}")
    model.save(f"{models_dir}/{TIMESTEPS*i}")       # save model every 10000 steps

# # Save the agent
# name = "dqn_lunar_" + str(datetime.datetime.now())
# model.save(name)
# del model  # delete trained model to demonstrate loading

# # Load the trained agent
# # NOTE: if you have loading issue, you can pass `print_system_info=True`
# # to compare the system on which the model was trained vs the current one
# # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
# model = DQN.load(name, env=env)

# # Evaluate the agent
# # NOTE: If you use wrappers with your environment that modify rewards,
# #       this will be reflected here. To evaluate with original rewards,
# #       wrap environment in a "Monitor" wrapper before other wrappers.
# # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)





# # episodes = 10 

# # for ep in range(episodes):
# #     obs = env.reset()
# #     done = False
# #     while not done:
# #         action, _states = model.predict(obs, deterministic=True)
# #         obs, rewards, dones, info = env.step(action)
# #         env.render()

# env.close()

# # for i in range(10000):
# #     action, _states = model.predict(obs, deterministic=True)
# #     obs, rewards, dones, info = env.step(action)
# #     env.render()