import gymnasium as gym
import shimmy
import numpy as np
from dm_control import suite
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC 
from gymnasium.wrappers import RecordVideo

# EVAL in seed number 10
seeds = [0, 1, 2]


for seed in seeds:

    model = SAC.load(f"weights/sac_cartpole_seed{seed}")
    env = gym.make("dm_control/cartpole-swingup-v0", render_mode="rgb_array")

    env = RecordVideo(env, video_folder=f"videos/trained_seed{seed}", name_prefix="eval", episode_trigger=lambda x: True)
    obs, _ = env.reset(seed=10)

    episode_reward = 0
    done = False
    truncated = False


    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward

    print(f"Seed {seed} | Evaluation Reward (Test Seed 10): {episode_reward}")

    env.close()



