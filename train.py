import gymnasium as gym
import shimmy
import numpy as np
from dm_control import suite
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC 



# env = gym.make("CartPole-v1", render_mode="human")

# env = suite.load(domain_name="cartpole", task_name="swingup")



seeds = [0, 1, 2]


for seed in seeds:
    env = gym.make("dm_control/cartpole-swingup-v0")
    env = Monitor(env, filename=f"logs/seed_{seed}")

    model = SAC("MultiInputPolicy", env, seed=seed, device="cuda:0", verbose=1)
    model.learn(total_timesteps=200000, log_interval=4)

    model.save(f"weights/sac_cartpole_seed{seed}")

    env.close()



