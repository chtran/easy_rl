from collections import defaultdict, namedtuple
import numpy as np
import gym
import click
import random

from utils.linear_estimator import LinearEstimator
from utils.cem_optimizer import CEMOptimizer
from agents import Agent


Observation = namedtuple('Observation', ['state', 'action', 'value'])

class DeepQLearningAgent(Agent):
    def __init__(self, 
            env, debug,
            discount=0.99, train_iters=50, 
            num_rollout_episodes=50):
        super().__init__(env, debug)
        self.action_space = env.action_space.shape[0] # |A|
        self.obs_space = env.observation_space.shape[0] # |S|
        self.q_estimator = LinearEstimator(n_in=self.action_space + self.obs_space, n_out=1)
        self.target_q_estimator = LinearEstimator(
            n_in=self.action_space + self.obs_space, n_out=1, W=self.q_estimator.W)

        self._discount = discount
        self._train_iters = train_iters
        self._num_rollout_episodes = num_rollout_episodes

    def get_action(self, state, explore_prob=0., return_value=False):
        if random.random() < explore_prob:
            return np.random.randn(self.action_space)
        fn = lambda action: self.target_q_estimator.predict(
            np.concatenate((state.flatten(), action)))
        best_action, value = CEMOptimizer(fn, self.action_space).optimize()

        if return_value:
            return best_action, value
        return best_action

    def train(self):
        explore_prob = 0.1
        for itr in range(self._train_iters):
            batch_data = []
            episode_rewards = []
            for epi_id in range(self._num_rollout_episodes):
                episode_data = []
                state = self.env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action = self.get_action(state, explore_prob=explore_prob)
                    next_state, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    episode_data.append((state, action, reward, next_state))
                    state = next_state
                    if self.debug and epi_id == 0:
                        self.env.render()
                episode_rewards.append(episode_reward)
                batch_data.extend(episode_data)
            print("Iter", itr, "Avg rewards:", np.mean(episode_rewards))

            self._train_batch(batch_data)
            print(self.q_estimator.W)
            explore_prob *= 0.9

    def _train_batch(self, batch_data):
        for state, action, reward, next_state in batch_data:
            best_next_action, next_value = self.get_action(next_state, return_value=True)
            target = reward + self._discount * next_value
            self.q_estimator.fit(np.concatenate((state.flatten(), action)), target)
        self.target_q_estimator.W = self.q_estimator.W
