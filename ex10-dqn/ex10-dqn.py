import random

import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import time


def build_NN(num_states, num_actions):
    """ Builds a neural network with 2 fully connected layers, input is the state and outputs are one value per action """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100, input_shape=(num_states,), activation="relu"))
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    model.add(tf.keras.layers.Dense(num_actions))
    return model


class DQN:
    def __init__(self):
        # environment:
        self.env = gym.make('CartPole-v1')
        self.num_states = len(self.env.observation_space.sample())
        self.num_actions = self.env.action_space.n
        # model:
        self.model = build_NN(self.num_states, self.num_actions)
        self.model.summary()
        # target_model:
        self.target_model = build_NN(self.num_states, self.num_actions)
        self.target_model.summary()

        # optimizer:
        # self.optimizer = tf.optimizers.SGD(learning_rate=0.01)
        self.optimizer = tf.optimizers.Adam()
        # parameters:
        self.discount = 0.99
        self.epsilon = 0.1
        self.buffer = []

    def Q_function(self, states, actions):
        """ This is the q-function approximated by the model, given state and action it outputs the value """
        return tf.reduce_sum(self.model(states) * tf.one_hot(actions, self.num_actions), axis=-1)

    @tf.function
    def gradient_step(self, states, action, value_target):
        """ Calculates the loss and applies the gradients """
        with tf.GradientTape() as tape:
            values = self.Q_function(states, action)
            loss = tf.reduce_mean(tf.square(value_target - values))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def train_episode(self):
        """ Runs one episode  """
        state = self.env.reset()
        done = False
        episode_reward = 0.
        N = 1
        K = 1
        while not done:
            # save target here
            self.target_model.set_weights(self.model.get_weights())
            for n in range(N + 1):
                if not done:
                    action = self.eps_greedy(state, self.epsilon)
                    next_state, reward, done, _ = self.env.step(action)
                    if done:
                        reward -= 100  # we apply an additional negative reward when done

                    self.buffer.append((state, action, reward, next_state))
                    episode_reward += reward
                    bkp_state = next_state

                for k in range(K + 1):
                    # sample from batch
                    state, action, reward, next_state = random.choice(self.buffer)

                    # compute the target value:
                    value_target = reward + self.discount * np.max(self.model(np.atleast_2d(next_state)), axis=-1)
                    # compute the loss and apply the gradients:
                    self.gradient_step(np.atleast_2d(state), action, value_target)

                state = bkp_state
        return episode_reward

    def eps_greedy(self, state, epsilon):
        """ epsilon greedy action selection """
        if np.random.rand() > epsilon:
            Q = self.target_model(np.atleast_2d(state))
            # Not a Number error
            # print(Q)
            # if np.flatnonzero(Q == np.max(Q)).size == 0:
            #    print("size 0")
            #    return np.random.choice(self.num_actions)
            return np.random.choice(np.flatnonzero(Q == np.max(Q)))  # randomly breaking ties
        return np.random.choice(self.num_actions)

    def run_policy(self):
        """ runs the current policy on the environment and renders it """
        state = self.env.reset()
        self.env.render()
        done = False
        while not done:
            action = self.eps_greedy(state, 0.)
            state, reward, done, _ = self.env.step(action)
            self.env.render()
            time.sleep(0.01)


def main():
    dqn = DQN()
    rewards = []
    for i in range(1000):
        reward = dqn.train_episode()
        rewards.append(reward)
        print(str(i) + " reward: " + str(reward))
    dqn.run_policy()
    plt.plot(rewards)
    plt.show()


if __name__ == '__main__':
    main()
