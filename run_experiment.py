#!/usr/bin/env python

from collections import defaultdict
import numpy as np
import gym
import click
import random
import tensorflow as tf

import agents

@click.command()
@click.argument("env_id", type=str, default="Point-v0")
@click.argument("agent_id", type=str, default="DDPG")
@click.option("--n_test_episodes", type=int, default=100)
@click.option("--render", type=bool, default=False)
@click.option("--debug", type=bool, default=False)
def main(env_id, agent_id, n_test_episodes, render, debug):
    rng = np.random.RandomState(42)

    if env_id == 'Point-v0':
        from environments import point_env
        env = gym.make('Point-v0')
    else:
        env = gym.make('MountainCarContinuous-v0')

    env.seed(42)
    tf.set_random_seed(42)
    np.random.seed(42)
    agent_class = agents.get_agent_class(agent_id)
    agent = agent_class(env, debug)
    agent.train()
    episode_rewards = []
    for i in range(n_test_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        render_episode = (i % 5 == 0)
        while not done:
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if render and render_episode:
                env.render()
        episode_rewards.append(episode_reward)
    print("Average rewards:", np.mean(episode_rewards))

if __name__ == "__main__":
    main()
