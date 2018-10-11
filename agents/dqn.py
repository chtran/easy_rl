from collections import namedtuple
import numpy as np
import tensorflow as tf
import gym
import click
import random

from utils.nn_estimator import NNEstimator
from utils.cem_optimizer import CEMOptimizer
from agents import Agent


class DeepQLearningAgent(Agent):
    def __init__(self, 
            env, debug,
            discount=0.99, train_iters=50, 
            num_rollout_episodes=50):
        super().__init__(env, debug)
        self.discrete = env.action_space.shape == ()
        self.obs_space = env.observation_space.shape[0] # |S|
        if self.discrete:
            self.action_space = env.action_space.n
        else:
            self.action_space = env.action_space.shape[0]

        self._discount = discount
        self._train_iters = train_iters
        self._num_rollout_episodes = num_rollout_episodes
        self.sess = tf.Session()
        self.sess.__enter__()
        self._build_graphs()
        tf.global_variables_initializer().run()


    def _build_graphs(self):
        if self.discrete:
            self.q_estimator = NNEstimator(
                n_in=self.obs_space, n_out=self.action_space,
                sess=self.sess, scope='q_estimator', hidden_dims=[10])
            self.target_q_estimator = NNEstimator(
                n_in=self.obs_space, n_out=self.action_space, 
                sess=self.sess, scope='target_q_estimator', hidden_dims=[10])
        else:
            self.q_estimator = NNEstimator(
                n_in=self.action_space + self.obs_space, n_out=1,
                sess=self.sess, scope='q_estimator')
            self.target_q_estimator = NNEstimator(
                n_in=self.action_space + self.obs_space, n_out=1, 
                sess=self.sess, scope='target_q_estimator')

        self.q_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, 
            scope=tf.get_variable_scope().name + "/q_estimator")
        self.target_q_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, 
            scope=tf.get_variable_scope().name + "/target_q_estimator")

    def get_action(self, state, explore_prob=0., return_value=False):
        if random.random() < explore_prob:
            return self._get_random_action()

        return self._get_best_action(state, return_value)

    def _get_random_action(self):
        if self.discrete:
            return np.random.randint(self.action_space)
        else:
            return np.random.randn(self.action_space)

    def _get_best_action(self, state, return_value):
        if self.discrete:
            action_values = self.target_q_estimator.predict(np.expand_dims(state, 0)) #N,A
            best_action = np.argmax(action_values, axis=1)
            if return_value:
                return best_action, action_values[best_action]
            return best_action
        else:
            fn = lambda action: self.target_q_estimator.predict(
                np.expand_dims(np.concatenate((state.flatten(), action)), 0))
            best_action, value = CEMOptimizer(fn, self.action_space).optimize()

            if return_value:
                return best_action, value
            return best_action

    def train(self):
        explore_prob = 0.1
        for itr in range(self._train_iters):
            batch_data = []
            episode_rewards = []
            print("Collecting trajectories")
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

            print("Training")
            self._train_batch(batch_data)
            self._update_target()
            explore_prob *= 0.9

    def _update_target(self):
        update_target_expr = []
        for var, var_target in zip(
                sorted(self.q_vars, key=lambda v: v.name),
                sorted(self.target_q_vars, key=lambda v: v.name)):
            update_target_expr.append(var.target.assign(var))
        return tf.group(*update_target_expr)

    def _train_batch(self, batch_data):
        # TODO: fit batch
        inputs = np.zeros((len(batch_data), self.obs_space + self.action_space))
        labels = np.zeros(len(batch_data))
        for i, row in enumerate(batch_data):
            state, action, reward, next_state = row
            best_next_action, next_value = self.get_action(next_state, return_value=True)
            target = reward + self._discount * next_value
            inputs[i, :] = np.concatenate((state.flatten(), action))
            labels[i, :] = target
        self.q_estimator.fit(inputs, labels)
