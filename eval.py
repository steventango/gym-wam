import argparse
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('message', type=str)
    args = parser.parse_args()
    env_id = 'WAMReachDense3DOF-v2'
    alg = TD3
    hidden_size = 32
    depth = 4
    model = alg.load(f"models/{env_id}_{alg.__name__}_{args.message}/best_model.zip")
    rewards = []
    episode_reward = 0

    env = gym.make(env_id, render_mode='human')
    observation, info = env.reset()
    while True:
        action, states_ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            observation, info = env.reset()
            rewards.append(episode_reward)
            print(f"Average Reward: {np.mean(rewards):.2f} +- {np.std(rewards):.2f}")
            episode_reward = 0

    env.close()


if __name__ == "__main__":
    main()
