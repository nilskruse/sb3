import gym

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.env_util import make_vec_env

env_name="LunarLanderContinuous-v2"
test_env = gym.make(env_name)
obs = test_env.reset()

model = PPO.load(f"{env_name}-PPO/checkpoints/rl_model_5040000_steps.zip")
#model = PPO.load(f"{env_name}-PPO/best_model.zip")

done = False;
total_reward = 0.0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    total_reward += reward
    test_env.render()
print(f"Reward is {total_reward}")
