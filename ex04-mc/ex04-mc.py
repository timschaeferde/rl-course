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


def main():
    # init
    env = gym.make('Blackjack-v0')
    # init V and rewards as dicts
    V = {}
    returns = {}

    for i in range(500000):
        # play one episode
        states, rewards = play(env)
        for s in states:
            G = rewards[states.index(s)]  # get reward of first occurrence
            if s in returns.keys():  # if already exists append else write with new key
                returns[s].append(G)
            else:
                returns[s] = [G]
            V[s] = np.mean(returns[s])
    # print(returns)
    print(V)
    # plot surface
    plot(V)


def V2Matrix(V: dict):
    # init matrices
    m1 = np.zeros((10, 10))
    m2 = np.zeros((10, 10))

    for key in V.keys():
        i = max(0, key[0] - 11) - 1  # index for card sum range 0-9 is 12-21
        j = key[1] - 1  # index dealer card from 0-9
        ace = key[2]
        if not i < 0:
            if ace:
                m1[i][j] = V[key]
            else:
                m2[i][j] = V[key]
    print(m1, m2)
    return m1, m2


def plot(V: dict):
    # generate two matrices from V
    m1, m2 = V2Matrix(V)

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
