from datetime import time, datetime

import gym
import numpy as np


def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    while not done:
        print("observation:", obs)
        if obs[0] >= 20:
            print("stick")
            obs, reward, done, _ = env.step(0)
        else:
            print("hit")
            obs, reward, done, _ = env.step(1)
        print("reward:", reward)
        print("")


class State:
    def __init__(self, s):
        self.player_sum = int(s[0])
        self.dealer_card = int(s[1])
        self.useable_ace = int(s[2])

    def tuple(self):
        return (self.player_sum, self.dealer_card, self.useable_ace)


class Q:
    def __init__(self, state_shape, action_shape):
        # dimension (player_sum, dealer_card, useable_ace, action_space)
        self.q_array = np.random.uniform(-1, 1, state_shape + (action_shape,))

    def get(self, s: State, a):
        return self.q_array[s.player_sum, s.dealer_card, s.useable_ace, a]

    def set(self, s: State, a, value):
        self.q_array[s.player_sum, s.dealer_card, s.useable_ace, a] = value

    def argmax(self, s: State, a):
        argmax = self.q_array[s.player_sum, s.dealer_card, s.useable_ace, :].argmax()
        return argmax


class Policy:
    def __init__(self, state_shape):
        # dimension (player_sum, dealer_card, useable_ace)
        self.pi_array = np.random.randint(0, 2, state_shape)
        # always hit below 12
        for i in range(12):
            self.pi_array[i, :, :] = 1

    def get(self, s: State):
        return self.pi_array[s.player_sum, s.dealer_card, s.useable_ace]

    def set(self, s: State, a):
        # always hit below 12
        if s.player_sum < 12:
            a = 1
        self.pi_array[s.player_sum, s.dealer_card, s.useable_ace] = a

    def print(self, usable_ace):
        # print and cut useless states like sum below 12...
        print(self.pi_array[12:22, 1:, int(usable_ace)])


class Returns:
    def __init__(self, state_shape, action_shape):
        # dimension (player_sum, dealer_card, useable_ace)
        self.sum_array = np.zeros(state_shape + (action_shape,))
        self.count_array = np.zeros(state_shape + (action_shape,))

    def get_avg(self, s: State, a):
        return self.sum_array[s.player_sum, s.dealer_card, s.useable_ace, a] / \
               self.count_array[s.player_sum, s.dealer_card, s.useable_ace, a]

    def append(self, s: State, a, value):
        self.sum_array[s.player_sum, s.dealer_card, s.useable_ace, a] += value
        self.count_array[s.player_sum, s.dealer_card, s.useable_ace, a] += 1


class EpisodeValues:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def sa(self):
        return zip(self.states, self.actions)


def monte_carlo_es():
    t0 = datetime.now()
    print(t0)
    env = gym.make('Blackjack-v0')
    state_shape = (32, 11, 2)
    action_shape = (2)
    q = Q(state_shape, action_shape)
    pi = Policy(state_shape)
    returns = Returns(state_shape, action_shape)

    eps = 0
    while eps < 1000000:
        eps += 1
        if eps % 100000 == 0:
            print("Finished {} episodes in {}.".format(eps, (datetime.now() - t0)))
            t0 = datetime.now()
            print("no usable ace")
            pi.print(usable_ace=False)
            print("usable ace")
            pi.print(usable_ace=True)
        # choose random start state
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        s = State(obs)
        # choose random start action
        a = int(env.action_space.sample())
        episode = EpisodeValues()
        done = False
        while not done:
            episode.states.append(s)  # append state
            episode.actions.append(a)  # append action
            obs, reward, done, _ = env.step(a)
            s = State(obs)
            episode.rewards.append(reward)  # append reward
            # choose new action from policy
            a = pi.get(s)

        for s, a in episode.sa():
            for i in range(len(episode.states)):
                if s is episode.states[i] and a == episode.actions[i]:
                    g = episode.rewards[i]
                    break
            returns.append(s, a, g)
            q.set(s, a, returns.get_avg(s, a))
        for s in episode.states:
            pi.set(s, q.argmax(s, a))


if __name__ == "__main__":
    # main()
    monte_carlo_es()
