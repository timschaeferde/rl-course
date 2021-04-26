import numpy as np
import matplotlib.pyplot as plt
import random


class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and covariance of 1.
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # init variables (rewards, n_plays, Q) by playing each arm once
    # init rewads and Q
    for arm in possible_arms:
        rewards[arm] = bandit.play_arm(arm)
        Q[arm] = rewards[arm] / n_plays [arm]
        n_plays[arm] += 1


    # Main loop
    while bandit.total_played < timesteps:
        # instead do greedy action selection
        arm = np.argmax(Q)  # select "best" action
        # update the variables (rewards, n_plays, Q) for the selected arm
        # play best action
        rewards[arm] += bandit.play_arm(arm)
        n_plays[arm] += 1
        # calculate normalized Q
        Q[arm] = rewards[arm] / n_plays[arm]




def epsilon_greedy(bandit, timesteps):
    # epsilon greedy action selection (you can copy your code for greedy as a starting point)
    eps = .1 # init epsilon

    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # init variables (rewards, n_plays, Q) by playing each arm once
    # init rewads and Q
    for arm in possible_arms:
        rewards[arm] = bandit.play_arm(arm)
        Q[arm] = rewards[arm] / n_plays [arm]
        n_plays[arm] += 1


    # Main loop
    while bandit.total_played < timesteps:
        # instead do greedy action selection
        arm = np.argmax(Q)  # select "best" action
        # update the variables (rewards, n_plays, Q) for the selected arm
        # play best action
        rewards[arm] += bandit.play_arm(arm)
        n_plays[arm] += 1
        # calculate normalized Q
        Q[arm] = rewards[arm] / n_plays[arm]


        if np.random.uniform() > eps: # decide whether to do greedy action or not
            arm = np.argmax(Q)  # select greedy action
        else:
            arm = random.choice(possible_arms) # select random action

            # Answer to 2 d): Here we could choose the second best for some probability as an additional strategy

        reward = bandit.play_arm(arm)  # play selected action
        n_plays[arm] += 1
        rewards[arm] += reward
        # calculate normalized Q
        Q[arm] = rewards[arm] / n_plays[arm]


def main():
    n_episodes = 10000
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print ("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.eps')
    plt.show()


if __name__ == "__main__":
    main()
