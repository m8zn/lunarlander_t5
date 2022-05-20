import gym
import matplotlib.pyplot as plt 
import time 

# env = gym.make("LunarLander-v2", continuous=True)
env = gym.make("LunarLander-v2")

obs_space = env.observation_space
act_space = env.action_space

print(obs_space)
print(act_space)

# Number of steps
num_steps = 1500

# matrix row: states column: actions
Q_ = np.zeros((4,4))


# reset the environment and see the initial observation
obs = env.reset()
print("The initial observation is {}".format(obs))

for step in range(num_steps):

    # select action
    a = np.argmax(Q_[])
    
    # Sample a random action from the entire action space
    random_action = env.action_space.sample()

    print(random_action)

    # constant_action = [0.01,0]

    # # Take the action and get the new observation space
    new_obs, reward, done, info = env.step(random_action)
    print("The new observation is {}".format(new_obs))  


    env.render()

    time.sleep(0.001)
    
    if done:
        env.reset()

    
env.close()

# import matplotlib.pyplot as plt 
# plt.imshow(env_screen)




# def demo_heuristic_lander(env, seed=None, render=False):

#     total_reward = 0
#     steps = 0
#     s = env.reset(seed=seed)
#     while True:
#         a = heuristic(env, s)
#         s, r, done, info = env.step(a)
#         total_reward += r

#         if render:
#             still_open = env.render()
#             if still_open is False:
#                 break

#         if steps % 20 == 0 or done:
#             print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
#             print(f"step {steps} total_reward {total_reward:+0.2f}")
#         steps += 1
#         if done:
#             break
#     if render:
#         env.close()
#     return total_reward


# if __name__ == "__main__":
# demo_heuristic_lander(LunarLander(), render=True)
