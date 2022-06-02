import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

env_name="LunarLander-v2"

run_dir = f"./{env_name}-PPO"
log_dir = f"{run_dir}/log"
checkpoint_dir = f"{run_dir}/checkpoints"
bestmodel_file = f"{run_dir}/best_model.zip"


# Parallel environments
env = make_vec_env(env_name, n_envs=4)

#callbacks
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoint_dir)

eval_env = make_vec_env(env_name, n_envs=1)
eval_callback = EvalCallback(eval_env, best_model_save_path=run_dir, log_path=run_dir, eval_freq=500)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=50000, callback=[checkpoint_callback, eval_callback], reset_num_timesteps=False)
model.save("some_model")
