from collections import defaultdict, namedtuple
import numpy as np
import gym
import click
import random

from utils.gaussian_estimator import GaussianEstimator
from utils.linear_estimator import LinearEstimator
from agents import Agent


Observation = namedtuple('Observation', ['state', 'action', 'value', 'baseline'])

class PolicyGradientAgent(Agent):
    def __init__(self, 
            env, debug,
            discount=0.99, train_iters=100, 
            learning_rate=0.1,
            num_rollout_episodes=50):
        super().__init__(env, debug)
        action_space = env.action_space.shape[0] # |A|
        obs_space = env.observation_space.shape[0] # |S|
        self.action_estimator = GaussianEstimator(obs_space, action_space)
        self.state_estimator = LinearEstimator(obs_space, 1)
        self._discount = discount
        self._train_iters = train_iters
        self._learning_rate = learning_rate
        self._num_rollout_episodes = num_rollout_episodes

    def get_action(self, state, explore_prob=0.):
        action = self.action_estimator.get_sample(state)
        return action

    def train(self):
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
        #print("Begin episode")
        for state, action, reward in episode_data[::-1]:
            value = reward + self._discount * next_value
            baseline = self.state_estimator.predict(np.square(state))
            #print(state, value)
            ret.append(Observation(state=state, action=action, value=value, baseline=baseline))
            next_value = value 
        #print("End episode")
        return ret

    def _train_batch(self, batch_data):
        # Fit actor
        action_grad = np.zeros_like(self.action_estimator.W)
        for obs in batch_data:
            action_grad += \
                self.action_estimator.get_grad_log_prob(obs.state, obs.action) * (obs.value -obs.baseline)
        action_grad = action_grad / (np.linalg.norm(action_grad) + 1e-8)
        self.action_estimator.W += self._learning_rate * action_grad

        # Fit critic
        for obs in batch_data:
            self.state_estimator.fit(np.square(obs.state), obs.value)
