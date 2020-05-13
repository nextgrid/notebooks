# # Create virtual display to render on remote machine
# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1, 1))
# display.start()

import matplotlib.pyplot as plt
# %matplotlib inline
from IPython import display
import gym
import numpy as np
from pathlib import Path

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import VecVideoRecorder
from stable_baselines.common.evaluation import evaluate_policy

import glob
import io
import base64
import IPython
from IPython.display import HTML


def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/', deterministic=True):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    eval_env.render(mode="human")
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()
    
## Display video
def show_videos(video_path='', prefix=''):
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>'''.format(mp4, video_b64.decode('ascii')))
    display.display(display.HTML(data="<br>".join(html)))
    
def record_and_show(env_id, model, name, folder="videos", deterministic=True, length=500):
    record_video(
        env_id, model, video_length=length, prefix=name, video_folder=folder, deterministic=deterministic
    )
    show_videos(folder, prefix=name)
    
def evaluate_model_vec(model, env, num_episodes=100, deterministic=True):
    """
    Multiprocess evaluation of ml-baselines RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param env: (SubprocVecEnv) vectorized environment 
        which defines how many cores will be used
    :param num_episodes: (int) minimal number of episodes to generate
    :param deterministic: (bool) make determenistic actions or not
    :return: (list) List of rewards for episodes
    """
    obs = env.reset()
    n_envs = len(env.envs)
    all_rewards = []
    episode_reward = np.zeros(n_envs)
    while len(all_rewards) < num_episodes:
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        # TODO: use numpy to find idx
        for i, finished in enumerate(done):
            if finished:
                all_rewards.append(episode_reward[i])
                episode_reward[i] = 0
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(all_rewards), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(all_rewards))
    return all_rewards