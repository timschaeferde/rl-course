import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
# random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
# env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def prob_sum(s, gamma, V_states):
    action_array = np.zeros(n_actions)
    # get prob for all actions
    for a in range(n_actions):
        P = env.P[s][a]
        # sum over all possible future sates
        for p in P:
            action_array[a] += p[0] * (p[2] + gamma * V_states[p[1]])
    return action_array


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm and return the policy
    steps = 0
    delta = np.inf
    while delta >= theta:
        steps += 1
        delta = 0
        # for all states calculate update the policy
        for s in range(n_states):
            v = V_states[s].copy()
            V_states[s] = np.max(prob_sum(s, gamma, V_states))
            # delta observes if the value function updated
            delta = max(delta, abs(v - V_states[s]))
    print("Number of steps: {}".format(steps))
    print("Optimal value function: {}".format(V_states))

    policy = np.zeros(n_states)
    # for all states get the best action
    for s in range(n_states):
        policy[s] = np.argmax(prob_sum(s, gamma, V_states))
    return policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
