from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
from hiive.mdptoolbox.example import forest
# import hiive_mdptoolbox.example
# import hiive_mdptoolbox
import gym
import numpy as np
import pandas
import sys
import os
from numpy.random import choice
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import time
np.random.seed(44)


P, R = forest(S=500, r1=10, r2= 16, p=0.1)

def test_policy(P, R, policy, test_count=1000, gamma=0.9):
    num_state = P.shape[-1]
    total_episode = num_state * test_count
    # start in each state
    total_reward = 0
    for state in range(num_state):
        state_reward = 0
        for state_episode in range(test_count):
            episode_reward = 0
            disc_rate = 1
            while True:
                # take step
                action = policy[state]
                # get next step using P
                probs = P[action][state]
                candidates = list(range(len(P[, color='red')ction][state])))
                next_state =  choice(candidates, 1, p=probs)[0]
                # get the reward
                reward = R[state][action] * disc_rate
                episode_reward += reward
                # when go back to 0 ended
                disc_rate *= gamma
                if next_state == 0:
                    break
            state_reward += episode_reward
        total_reward += state_reward
    return total_reward / total_episode

def trainVI(P, R, discount=0.9, epsilon=[1e-9]):
    vi_df = pd.DataFrame(columns=["Epsilon", "Policy", "Iteration", 
                                  "Time", "Reward", "Value Function"])
    for eps in epsilon:
        vi = ValueIteration(P, R, gamma=discount, epsilon=eps, max_iter=100000)
        vi.run()
        reward = test_policy(P, R, vi.policy)
        info = [float(eps), vi.policy, vi.iter, vi.time, reward, vi.V]
        df_length = len(vi_df)
        vi_df.loc[df_length] = info
    return vi_df

def value_iteration(P, R):
    print('VI')
    gamma = 0.9
    epsilons = [1e-20, 1e-15, 1e-10, 1e-05, 1e-01]
    scores = []
    times = []
    iterations = []
    for epsilon in epsilons:
        startTime = time.time()
        vi = ValueIteration(P, R, gamma=gamma, epsilon=epsilon, max_iter=100000)
        vi.run()
        endTime = time.time()
        t = endTime - startTime
        reward = test_policy(P, R, vi.policy)
        scores.append(reward)
        times.append(t)
        iterations.append(vi.iter)
        print(f'Best score: {reward}, Iterations: {vi.iter}, Time: {t}')

    df = pandas.DataFrame(data={
        'epsilon': epsilons,
        'iterations': iterations,
        'times': times,
        'scores': scores
    }, index=epsilons)
    fig, ax = plt.subplots(1, 3, figsize=(24, 5))
    fig.suptitle('Policy Iteration - Epsilon Analysis (gamma=1)')
    df['iterations'].plot(ax=ax[0], xlabel='epsilon', ylabel='iterations', logx=True, color='red')
    df['scores'].plot(ax=ax[1], xlabel='epsilon', ylabel='reward', logx=True, color='red')
    df['times'].plot(ax=ax[2], xlabel='epsilon', ylabel='seconds', logx=True, color='red')
    plt.savefig('figures/forest_value_iteration_epsilon.png')

    gammas = [0.8, 0.9, 0.95, 0.99]
    epsilon = 1e-20
    scores = []
    times = []
    iterations = []
    for gamma in gammas:
        startTime = time.time()
        vi = ValueIteration(P, R, gamma=gamma, epsilon=epsilon, max_iter=100000)
        vi.run()
        endTime = time.time()
        t = endTime - startTime
        reward = test_policy(P, R, vi.policy)
        scores.append(reward)
        times.append(t)
        iterations.append(vi.iter)
        print(f'Best score: {reward}, Iterations: {vi.iter}, Time: {t}')

    df = pandas.DataFrame(data={
        'gammas': gamma,
        'iterations': iterations,
        'times': times,
        'scores': scores
    }, index=gammas)
    fig, ax = plt.subplots(1, 3, figsize=(24, 5))
    fig.suptitle('Value Iteration - Gamma Analysis (epsilon=1e-20)')
    df['iterations'].plot(ax=ax[0], xlabel='gamma', ylabel='iterations', color='red')
    df['scores'].plot(ax=ax[1], xlabel='gamma', ylabel='reward', color='red')
    df['times'].plot(ax=ax[2], xlabel='gamma', ylabel='seconds', color='red')
    plt.savefig('figures/forest_value_iteration_gamma.png')


def policy_iteration(P, R):
    print('PI')
    gammas = [0.8, 0.9, 0.95, 0.99]
    scores = []
    times = []
    iterations = []
    for gamma in gammas:
        startTime = time.time()
        pi = PolicyIteration(P, R, gamma=gamma, max_iter=1e6)
        pi.run()
        endTime = time.time()
        pi_pol = pi.policy
        pi_reward = test_policy(P, R, pi_pol)
        pi_iter = pi.iter
        t = endTime - startTime
        scores.append(pi_reward)
        times.append(t)
        iterations.append(pi_iter)
        print(f'Best score: {pi_reward}, Iterations: {pi_iter}, Time: {t}')

    df = pandas.DataFrame(data={
        'gammas': gammas,
        'iterations': iterations,
        'times': times,
        'scores': scores
    }, index=gammas)
    fig, ax = plt.subplots(1, 3, figsize=(24, 5))
    fig.suptitle('Policy Iteration - Gamma Analysis (epsilon=1e-20)')
    df['iterations'].plot(ax=ax[0], xlabel='gamma', ylabel='iterations', color='red')
    df['scores'].plot(ax=ax[1], xlabel='gamma', ylabel='reward', color='red')
    df['times'].plot(ax=ax[2], xlabel='gamma', ylabel='seconds', color='red')
    plt.savefig('figures/forest_policy_iteration_gamma.png')


def q_learn(P ,R):
    print('QL')
    scores = []
    times = []
    iterations = []
    alphas = [0.01, 0.05, 0.1]
    for alpha in alphas:
        startTime = time.time()
        q = QLearning(P, R, gamma=0.9, alpha=alpha, n_iter=10000)
        q.run()
        endTime = time.time()
        t = endTime - startTime
        reward = test_policy(P, R, q.policy)
        scores.append(reward)
        times.append(t)
        print(f'Best score: {reward}, Time: {t}')
    df = pandas.DataFrame(data={
        'alphas': alphas,
        'times': times,
        'scores': scores
    }, index=alphas)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Q-Learning - Initial Alpha Analysis (gamma=0.9)')
    df['scores'].plot(ax=ax[0], xlabel='initial alpha', ylabel='reward', color='red')
    df['times'].plot(ax=ax[1], xlabel='initial alpha', ylabel='seconds', color='red')
    plt.savefig('figures/forest_qlearning_alpha.png')

    scores = []
    times = []
    gammas = [0.9, 0.95, 0.99, 0.999]
    for gamma in gammas:
        startTime = time.time()
        q = QLearning(P, R, gamma=gamma, alpha=0.1, n_iter=10000)
        q.run()
        endTime = time.time()
        t = endTime - startTime
        reward = test_policy(P, R, q.policy)
        scores.append(reward)
        times.append(t)
        print(f'Best score: {reward}, Time: {t}')
    df = pandas.DataFrame(data={
        'gammas': gammas,
        'times': times,
        'scores': scores
    }, index=gammas)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Q-Learning - Gamma Analysis (alpha=0.1)')
    df['scores'].plot(ax=ax[0], xlabel='gamma', ylabel='reward', ylim=[0, 1], color='red')
    df['times'].plot(ax=ax[1], xlabel='gamma', ylabel='seconds', color='red')
    plt.savefig('figures/forest_qlearning_gamma.png')

value_iteration(P, R)
# policy_iteration(P, R)
# q_learn(P, R)