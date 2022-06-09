import gym
import time
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import os
import time

from os import listdir
from os.path import isfile, join


env_name = "LunarLander-v2"

run_name = f"{env_name}-v1"
run_dir = f"./{env_name}-PPO"
log_dir = f"{run_dir}/log"
checkpoint_dir = f"{run_dir}/checkpoints"
bestmodel_file = f"{run_dir}/best_model.zip"

NUM_CPU = 12

env = make_vec_env(env_name, n_envs=NUM_CPU)

eval_env = make_vec_env(env_name, n_envs=NUM_CPU)
eval_callback = EvalCallback(eval_env, best_model_save_path=run_dir, log_path=run_dir, eval_freq=2000, n_eval_episodes = 10)

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoint_dir, name_prefix=run_name)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=1000000, callback=[checkpoint_callback, eval_callback], reset_num_timesteps=False, tb_log_name=run_name)
