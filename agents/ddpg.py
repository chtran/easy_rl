from collections import defaultdict
import numpy as np
import gym
import click
import random

from utils.linear_estimator import LinearEstimator
from agents import Agent


class DDPGAgent(Agent):
    def __init__(self, 
            env, 
            debug,
            discount=0.99, train_iters=20, batch_train_iters=10, 
            batch_size=10000, use_replay_buffer=True):
        super().__init__(env, debug)
        self.actor = ActorEstimator(env)
        self.critic = CriticEstimator(env)

        self._discount = discount
        self._train_iters = train_iters
        self._batch_train_iters = batch_train_iters
        self._batch_size = batch_size
        self._use_replay_buffer = use_replay_buffer

    def get_action(self, state, explore_prob=0.):
        action = self.actor.get_action(state, explore_prob)
        return action

    def train(self):
        # Policy training loop
        data = []
        for itr in range(self._train_iters):
            # Collect trajectory loop
            batch_data = []
            episode_rewards = []

            explore_prob = 0.5*(0.9**itr)
            print("Explore prob:", explore_prob)
            while len(batch_data) < self._batch_size:
                state = self.env.reset()
                done = False
                # Only render the first trajectory
                render_episode = len(batch_data) == 0
                episode_reward = 0
                # Collect a new trajectory
                while not done:
                    action = self.get_action(state, explore_prob=explore_prob)
                    next_state, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    batch_data.append((state, action, reward, next_state))
                    state = next_state
                    if self.debug and render_episode:
                        self.env.render()
                episode_rewards.append(episode_reward)
            data.extend(batch_data)
            if self._use_replay_buffer:
                random.shuffle(data)
            self._train_batch(data[-self._batch_size:])
            print(self.actor.target_estimator.W)
            print(self.critic.target_estimator.W)
            norm = np.linalg.norm(self.actor.estimator.W)
            print("Iter", itr, "Avg rewards:", np.mean(episode_rewards), "Norm:", norm)

    def _train_batch(self, data):
        for i in range(self._batch_train_iters):
            for state, action, reward, next_state in data:
                predicted_next_action = self.actor.get_action(next_state)
                target = reward + self._discount * self.critic.evaluate(next_state, predicted_next_action)
                action_grad = -self.critic.action_grad(action)
                self.actor.estimator.fit_to_delta(
                    state, action_grad)
                self.critic.fit(state, action, target)
            self.actor.soft_update()
            self.critic.soft_update()


class ActorEstimator(object):
    def __init__(self, env, soft_update_weight=0.9):
        self.estimator = LinearEstimator(obs_space, action_space)
        self.target_estimator = LinearEstimator(
            obs_space, action_space, W=self.estimator.W)

        self._soft_update_weight = soft_update_weight
        self._action_lb = env.action_space.low # |A|
        self._action_ub = env.action_space.high # |A|
    
    def soft_update(self):
        self.target_estimator.W += self._soft_update_weight * (self.estimator.W - self.target_estimator.W)

    def get_action(self, state, explore_prob=0.):
        """Return dim: |A|"""
        add_noise = random.random() < explore_prob
        action = self.target_estimator.predict(state, add_noise)
        action = np.clip(action, self.action_lb, self.action_ub)
        return action
    

class CriticEstimator(object):
    def __init__(self, env, soft_update_weight=0.9):
        self.estimator = LinearEstimator(self.obs_space + self.action_space, 1)
        self.target_estimator = LinearEstimator(
            self.obs_space + self.action_space, 1, W=self.estimator.W)

        self._soft_update_weight = soft_update_weight
        self._action_space = env.action_space.shape[0] # |A|
        self._obs_space = env.observation_space.shape[0] # |S|

    def soft_update(self):
        self.target_estimator.W += self._soft_update_weight * (self.estimator.W - self.target_estimator.W)

    def evaluate(self, state, action):
        return self.target_estimator.predict(np.concatenate((state.flatten(), action)))

    def action_grad(self, action):
        # |A|
        grad = self.estimator.partial_grad(self._obs_space, self._obs_space + self._action_space)
        return grad

    def fit(self, state, action, target):
        x = np.concatenate((state.flatten(), action))
        self.estimator.fit(x, target)
