import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

model = PPO.load("ppo_cartpole")
test_env = gym.make("CartPole-v1")
obs = test_env.reset()
done = False;

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    test_env.render()
