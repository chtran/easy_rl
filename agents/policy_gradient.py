from collections import defaultdict
import numpy as np
import gym
import click
import random

from utils.linear_estimator import LinearEstimator
from agents import Agent


class PolicyGradientAgent(Agent):
    def __init__(self, 
            env, 
            debug,
            discount=0.99, train_iters=20, batch_train_iters=10, 
            batch_size=10000, use_replay_buffer=True):
        super().__init__(env, debug)
        self.discount = discount
        self.train_iters = train_iters
        self.batch_train_iters = batch_train_iters
        self.batch_size = batch_size
        self.use_replay_buffer = use_replay_buffer

    def get_action(self, state, explore_prob=0.):
        action = self.estimator.get_action(state)
        return action

    def train(self):
        # Policy training loop
        data = []
        for itr in range(self.train_iters):
            # Collect trajectory loop
            batch_data = []
            episode_rewards = []

            explore_prob = 0.5*(0.9**itr)
            print("Explore prob:", explore_prob)
            while len(batch_data) < self.batch_size:
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
            if self.use_replay_buffer:
                random.shuffle(data)
            self._train_batch(data[-self.batch_size:])
            print(self.actor.target_estimator.W)
            print(self.critic.target_estimator.W)
            norm = np.linalg.norm(self.actor.estimator.W)
            print("Iter", itr, "Avg rewards:", np.mean(episode_rewards), "Norm:", norm)

    def _train_batch(self, data):
        for i in range(self.batch_train_iters):
            for state, action, reward, next_state in data:
                predicted_next_action = self.actor.get_action(next_state)
                target = reward + self.discount * self.critic.evaluate(next_state, predicted_next_action)
                action_grad = -self.critic.action_grad(action)
                self.actor.estimator.fit_to_delta(
                    state, action_grad)
                self.critic.fit(state, action, target)
            self.actor.soft_update()
            self.critic.soft_update()



