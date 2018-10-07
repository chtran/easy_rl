from collections import defaultdict, namedtuple
import numpy as np
import gym
import click
import random

from utils.gaussian_estimator import GaussianEstimator
from agents import Agent


Observation = namedtuple('Observation', ['state', 'action', 'value'])

class PolicyGradientAgent(Agent):
    def __init__(self, 
            env, debug,
            discount=0.99, train_iters=20, 
            learning_rate=0.1,
            num_rollout_episodes=50):
        super().__init__(env, debug)
        action_space = env.action_space.shape[0] # |A|
        obs_space = env.observation_space.shape[0] # |S|
        self.estimator = GaussianEstimator(obs_space, action_space)
        self._discount = discount
        self._train_iters = train_iters
        self._learning_rate = learning_rate
        self._num_rollout_episodes = num_rollout_episodes

    def get_action(self, state, explore_prob=0.):
        action = self.estimator.get_sample(state)
        return action

    def train(self):
        print(self.debug)
        for itr in range(self._train_iters):
            batch_data = []
            episode_rewards = []
            for epi_id in range(self._num_rollout_episodes):
                episode_data = []
                state = self.env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action = self.get_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    episode_data.append((state, action, reward))
                    state = next_state
                    if self.debug and epi_id == 0:
                        self.env.render()
                episode_rewards.append(episode_reward)
                batch_data.extend(self._process_episode_data(episode_data))
            print("Iter", itr, "Avg rewards:", np.mean(episode_rewards))

            self._train_batch(batch_data)

    def _process_episode_data(self, episode_data):
        ret = []
        next_value = 0
        for state, action, reward in episode_data[::-1]:
            value = reward + self._discount * next_value
            ret.append(Observation(state=state, action=action, value=value))
            next_value = value 
        return ret

    def _train_batch(self, batch_data):
        grad = np.zeros_like(self.estimator.W)
        for obs in batch_data:
            grad += self.estimator.get_grad_log_prob(obs.state, obs.action) * obs.value
        grad = grad / (np.linalg.norm(grad) + 1e-8)
        self.estimator.W += self._learning_rate * grad
