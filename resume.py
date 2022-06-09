import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from os import listdir
from os.path import isfile, join


env_name="Taxi-v3"

run_dir = f"./{env_name}-PPO"
log_dir = f"{run_dir}/log"
checkpoint_dir = f"{run_dir}/checkpoints"
bestmodel_file = f"{run_dir}/best_model.zip"

checkpoint_prefix = "rl_model_";
checkpoint_postfix = "_steps.zip";

onlyfiles = [int(f.lstrip(checkpoint_prefix).rstrip(checkpoint_postfix)) for f in listdir(checkpoint_dir) if isfile(join(checkpoint_dir, f))]
onlyfiles.sort(reverse=True)

print(onlyfiles)
print(f"First file is {onlyfiles[0]}")

latest_checkpoint = checkpoint_prefix + str(onlyfiles[0]) + checkpoint_postfix

print(f"Latest checkpoint file is {latest_checkpoint}")
# Parallel environments
env = make_vec_env(env_name, n_envs=12)

#callbacks
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoint_dir)

eval_env = make_vec_env(env_name, n_envs=1)
eval_callback = EvalCallback(eval_env, best_model_save_path=run_dir, log_path=run_dir, eval_freq=2000 * 12, n_eval_episodes = 10)

model = PPO.load(f"{checkpoint_dir}/{latest_checkpoint}", env=env)

#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=1000000, callback=[checkpoint_callback, eval_callback], reset_num_timesteps=False)
#model.save("some_model")
