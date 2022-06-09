import gym
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

env_name="CarRacing-v1"

run_dir = f"./{env_name}-PPO"
log_dir = f"{run_dir}/log"
checkpoint_dir = f"{run_dir}/checkpoints"
bestmodel_file = f"{run_dir}/best_model.zip"


# Parallel environments
env = make_vec_env(env_name, n_envs=12)

#callbacks
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir)

eval_env = make_vec_env(env_name, n_envs=12)
eval_callback = EvalCallback(eval_env, best_model_save_path=run_dir, log_path=run_dir, eval_freq=2000, n_eval_episodes = 10)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=10000000, callback=[checkpoint_callback, eval_callback], reset_num_timesteps=False)
