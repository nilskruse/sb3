import gym

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.env_util import make_vec_env
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

checkpoint_prefix = f"{run_name}_";
checkpoint_postfix = "_steps.zip";

onlyfiles = [int(f.lstrip(checkpoint_prefix).rstrip(checkpoint_postfix)) for f in listdir(checkpoint_dir) if isfile(join(checkpoint_dir, f)) and run_name in f]
onlyfiles.sort(reverse=True)

print(onlyfiles)
print(f"First file is {onlyfiles[0]}")

latest_checkpoint = checkpoint_prefix + str(onlyfiles[0]) + checkpoint_postfix

print(f"Latest checkpoint file is {latest_checkpoint}")
# Parallel environments

env = gym.make(env_name)

#callbacks


model_path = f"{checkpoint_dir}/{latest_checkpoint}"

model = PPO.load(model_path, env=env)

done = False;
obs = env.reset()
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
