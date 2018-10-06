#!/usr/bin/env python

from collections import defaultdict
import numpy as np
import gym
import click
import random


class DDPGAgent:
    def __init__(self, env, discount=0.99, train_iters=10):
        self.actor = ActorEstimator(env)
        self.critic = CriticEstimator(env)
        self.discount = discount
        self.train_iters = train_iters

    def get_action(self, state, explore_prob=0.):
        action = self.actor.get_action(state, explore_prob)
        return action

    def train(self, data):
        for i in range(train_iters):
            for state, action, reward, next_state in data:
                predicted_next_action = self.actor.get_action(next_state)
                target = reward + self.discount * self.critic.evaluate(next_state, predicted_next_action)
                action_grad = -self.critic.action_grad(action)
                self.actor.estimator.fit_to_delta(
                    state, action_grad)
                self.critic.fit(state, action, target)
            self.actor.soft_update()
            self.critic.soft_update()


class ActorEstimator(object):
    def __init__(self, env):
        action_space = env.action_space.shape[0] # |A|
        obs_space = env.observation_space.shape[0] # |S|
        self.action_lb = env.action_space.low # |A|
        self.action_ub = env.action_space.high # |A|
        self.estimator = LinearEstimator(obs_space, action_space)
        self.target_estimator = LinearEstimator(
            obs_space, action_space, W=self.estimator.W)
    
    def soft_update(self):
        self.target_estimator.W += 1.0 * (self.estimator.W - self.target_estimator.W)

    def get_action(self, state, explore_prob=0.):
        """Return dim: |A|"""
        add_noise = random.random() < explore_prob
        action = self.target_estimator.predict(state, add_noise)
        action = np.clip(action, self.action_lb, self.action_ub)
        return action
    

class CriticEstimator(object):
    def __init__(self, env):
        self.action_space = env.action_space.shape[0] # |A|
        self.obs_space = env.observation_space.shape[0] # |S|
        self.estimator = LinearEstimator(self.obs_space + self.action_space, 1)
        self.target_estimator = LinearEstimator(
            self.obs_space + self.action_space, 1, W=self.estimator.W)

    def soft_update(self):
        self.target_estimator.W += 0.9 * (self.estimator.W - self.target_estimator.W)

    def evaluate(self, state, action):
        return self.target_estimator.predict(np.concatenate((state.flatten(), action)))

    def action_grad(self, action):
        # |A|
        grad = self.estimator.partial_grad(self.obs_space, self.obs_space + self.action_space)
        return grad

    def fit(self, state, action, target):
        lets_print = random.random() < 0.001
        x = np.concatenate((state.flatten(), action))
        self.estimator.fit(x, target)


@click.command()
@click.argument("env_id", type=str, default="Point-v0")
@click.option("--batch_size", type=int, default=2000)
@click.option("--discount", type=float, default=0.99)
@click.option("--learning_rate", type=float, default=0.1)
@click.option("--n_itrs", type=int, default=100)
@click.option("--render", type=bool, default=False)
@click.option("--use_replay_buffer", type=bool, default=False)
def main(
        env_id, batch_size, discount, 
        learning_rate, n_itrs, render, 
        use_replay_buffer):
    rng = np.random.RandomState(42)

    if env_id == 'MountainCarContinuous-v0':
        env = gym.make('MountainCarContinuous-v0')
    elif env_id == 'Point-v0':
        from simplepg import point_env
        env = gym.make('Point-v0')
    else:
        raise ValueError(
            "Unsupported environment: must be one of 'MountainCarContinuous-v0', 'Point-v0'")

    env.seed(42)
    agent = DDPGAgent(env, discount)
    data = []

    # Policy training loop
    for itr in range(n_itrs):
        # Collect trajectory loop
        batch_data = []
        episode_rewards = []

        explore_prob = 0.5*(0.9**itr)
        print("Explore prob:", explore_prob)
        while len(batch_data) < batch_size:
            state = env.reset()
            done = False
            # Only render the first trajectory
            render_episode = len(batch_data) == 0
            episode_reward = 0
            # Collect a new trajectory
            while not done:
                action = agent.get_action(state, explore_prob=explore_prob)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                batch_data.append((state, action, reward, next_state))
                state = next_state
                if render and render_episode:
                    env.render()
            episode_rewards.append(episode_reward)
        data.extend(batch_data)
        if use_replay_buffer:
            random.shuffle(data)
        agent.train(data[-batch_size:])
        print(agent.actor.target_estimator.W)
        print(agent.critic.target_estimator.W)
        norm = np.linalg.norm(agent.actor.estimator.W)
        print("Iter", itr, "Avg rewards:", np.mean(episode_rewards), "Norm:", norm)

if __name__ == "__main__":
    main()
