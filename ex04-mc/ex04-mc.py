from copy import copy

import gym
import numpy as np
from matplotlib import pyplot as plt


def play(env):
    # print("\nNew game!\n")
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    states = []
    rewards = []
    done = False
    while not done:
        # print("observation:", obs)
        # observations are our states
        states.append(obs)
        if obs[0] >= 20:
            # print("stick")
            obs, reward, done, _ = env.step(0)
        else:
            # print("hit")
            obs, reward, done, _ = env.step(1)
        rewards.append(reward)
        # print("reward:", reward)
        # print("")
    return states, rewards


def monte_carlo_first():
    # init
    env = gym.make('Blackjack-v0')
    # init V and rewards as dicts
    v = {}
    returns = {}
    eps = 0
    while eps < 10000:
        eps += 1
        # play one episode
        states, rewards = play(env)
        for s in states:
            G = rewards[states.index(s)]  # get reward of first occurrence
            if s in returns.keys():  # if already exists append else write with new key
                returns[s].append(G)
            else:
                returns[s] = [G]  # init new key and set list
    # moved out of loop for faster computation
    for s in returns.keys():
        v[s] = np.mean(returns[s])
    # print(returns)
    print(v)
    # plot surface
    plot(v)


def init_pi(env):
    pi = {}
    for s0 in range(env.observation_space[0].n):
        for s1 in range(env.observation_space[1].n):
            for s2 in range(env.observation_space[2].n):
                pi[(s0, s1, s2)] = env.action_space.sample()
    return pi


def monte_carlo_es():
    # init
    env = gym.make('Blackjack-v0')
    # init V and rewards as dicts
    q = {}
    pi = init_pi(env)
    print(pi)
    returns = {}

    eps = 0
    while eps < 1000000:
        eps += 1
        if eps % 100 == 0:
            print(pi)
        # choose action and state
        a = env.action_space.sample()
        obs = env.reset()

        # play env
        states_actions = []
        rewards = []
        done = False

        while not done:
            # observations are our states
            states_actions.append((obs, a))
            # first step
            obs, reward, done, _ = env.step(a)
            # override after new obs
            a = pi[obs]
            rewards.append(reward)

        for s_a in states_actions:
            G = rewards[states_actions.index(s_a)]  # get reward of first occurrence
            if s_a in returns.keys():  # if already exists append else write with new key
                returns[s_a].append(G)
            else:
                returns[s_a] = [G]  # init new key and set list
        #print(returns)
        # moved out of loop for faster computation
        for s_a in returns.keys():
            mean = np.mean(returns[s_a])
            #print(s_a,mean)
            q[s_a] = mean
        for s_a in q.keys():
            s = s_a[0]
            actions = [0, 0]
            for a in range(env.action_space.n):
                if (s, a) in q.keys():
                    actions[a] = q[(s, a)]
            pi[s] = np.argmax(actions)
        #print(pi)

def main():
    # monte_carlo_first()
    monte_carlo_es()


def V2Matrix(v: dict):
    # init matrices
    m1 = np.zeros((10, 10))
    m2 = np.zeros((10, 10))

    for key in v.keys():
        i = max(0, key[0] - 11) - 1  # index for card sum range 0-9 is 12-21
        j = key[1] - 1  # index dealer card from 0-9
        ace = key[2]
        if not i < 0:
            if ace:
                m1[i][j] = v[key]
            else:
                m2[i][j] = v[key]
    # print(m1, m2)
    return m1, m2


def plot(v: dict):
    # generate two matrices from V
    m1, m2 = V2Matrix(v)

    # plot each matrix as surface
    (fig, ax, surf) = surface_plot(m1, cmap=plt.cm.coolwarm)
    plt.title("with usable ace")
    ax.set_xlabel('X (dealer_card)')
    ax.set_ylabel('Y (player_sum)')
    ax.set_zlabel('Z (values)')

    plt.show()

    (fig, ax, surf) = surface_plot(m2, cmap=plt.cm.coolwarm)
    plt.title("no usable ace")
    ax.set_xlabel('X (dealer_card)')
    ax.set_ylabel('Y (player_sum)')
    ax.set_zlabel('Z (values)')

    plt.show()


def surface_plot(matrix, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)


if __name__ == "__main__":
    main()
