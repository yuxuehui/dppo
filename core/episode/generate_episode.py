import gym
import numpy as np
from stable_baselines3 import PPO
from core.utils.env import make_vector_env
import os

# 假设你已经训练好了多个agents，并且它们保存在以下路径
agent_paths = {
    'Breakout': 'path_to_breakout_agent.zip',
    'Pong': 'path_to_pong_agent.zip',
    # 添加更多的游戏和对应的agent路径
}


# 生成数据的函数
def generate_episode_data(env_name, agent_path, num_envs=10, num_episodes=100):
    #env = gym.make(env_name)
    env = make_vector_env(env_name, num_envs,
                    normalization_path=os.path.join(self.cfg.train.save_path, "vecnormalize.pkl"))
    model = PPO.load(agent_path)

    episodes_data = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode = []

        while not done:
            action, _states = model.predict(obs)
            new_obs, reward, done, info = env.step(action)
            episode.append((obs, action, reward, new_obs))
            obs = new_obs

        episodes_data.append(episode)

    env.close()
    return episodes_data


# 保存数据到文件
import pickle

for game, path in agent_paths.items():
    data = generate_episode_data(game, path)
    with open(f'{game}_episodes.pkl', 'wb') as f:
        pickle.dump(data, f)
