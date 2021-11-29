from gym.envs.toy_text.frozen_lake import generate_random_map
import gym
import numpy
import pandas
import time
from matplotlib import pyplot as plt
from model_based import valueIteration, calculatePolicy, evaluatePolicy, policyIteration, q_learning, test_qpolicy
from q_learning import QLearner

numpy.random.seed(42)

# random_map = generate_random_map(size=30, p=0.9)
random_map = None
env = gym.make('FrozenLake-v1', desc=random_map)
env.seed(42)

def value_iteration(env):
    gamma = 0.990
    epsilons = [1e-20, 1e-15, 1e-10, 1e-05, 1e-01]
    scores = []
    times = []
    iterations = []
    for epsilon in epsilons:
        startTime = time.time()
        optimal_V, it, statistics = valueIteration(env, gamma, epsilon)
        optimal_policy = calculatePolicy(env, optimal_V, gamma)
        endTime = time.time()
        score = evaluatePolicy(env, optimal_policy, gamma, 1000)
        t = endTime - startTime
        scores.append(score)
        times.append(t)
        iterations.append(it)
        print(f'Best score: {score}, Iterations: {it}, Time: {endTime-startTime}')

    df = pandas.DataFrame(data={
        'epsilon': epsilons,
        'iterations': iterations,
        'times': times,
        'scores': scores
    }, index=epsilons)
    fig, ax = plt.subplots(1, 3, figsize=(24, 5))
    fig.suptitle('Value Iteration - Epsilon Analysis (gamma=1)')
    df['iterations'].plot(ax=ax[0], xlabel='epsilon', ylabel='iterations', logx=True)
    df['scores'].plot(ax=ax[1], xlabel='epsilon', ylabel='reward', logx=True, ylim=[0, 1])
    df['times'].plot(ax=ax[2], xlabel='epsilon', ylabel='seconds', logx=True)
    plt.savefig('figures/frozen_lake_value_iteration_epsilon.png')

    gammas = [0.9, 0.95, 0.99, 0.999]
    epsilon = 1e-20
    scores = []
    times = []
    iterations = []
    for gamma in gammas:
        startTime = time.time()
        optimal_V, it, statistics = valueIteration(env, gamma, epsilon)
        optimal_policy = calculatePolicy(env, optimal_V, gamma)
        endTime = time.time()
        score = evaluatePolicy(env, optimal_policy, gamma, 1000)
        t = endTime - startTime
        scores.append(score)
        times.append(t)
        iterations.append(it)
        print(f'Best score: {score}, Iterations: {it}, Time: {endTime-startTime}')

    df = pandas.DataFrame(data={
        'gammas': gamma,
        'iterations': iterations,
        'times': times,
        'scores': scores
    }, index=gammas)
    fig, ax = plt.subplots(1, 3, figsize=(24, 5))
    fig.suptitle('Value Iteration - Gamma Analysis (epsilon=1e-20)')
    df['iterations'].plot(ax=ax[0], xlabel='gamma', ylabel='iterations')
    df['scores'].plot(ax=ax[1], xlabel='gamma', ylabel='reward', ylim=[0, 1])
    df['times'].plot(ax=ax[2], xlabel='gamma', ylabel='seconds')
    plt.savefig('figures/frozen_lake_value_iteration_gamma.png')


def policy_iteration(env):
    gamma = 0.999
    epsilons = [1e-20, 1e-15, 1e-10, 1e-05, 1e-01]
    scores = []
    times = []
    iterations = []
    for epsilon in epsilons:
        startTime = time.time()
        optimal_policy, it = policyIteration(env, gamma, epsilon)
        endTime = time.time()
        score = evaluatePolicy(env, optimal_policy, gamma, 1000)
        t = endTime - startTime
        scores.append(score)
        times.append(t)
        iterations.append(it)
        print(f'Best score: {score}, Iterations: {it}, Time: {endTime-startTime}')

    df = pandas.DataFrame(data={
        'epsilon': epsilons,
        'iterations': iterations,
        'times': times,
        'scores': scores
    }, index=epsilons)
    fig, ax = plt.subplots(1, 3, figsize=(24, 5))
    fig.suptitle('Policy Iteration - Epsilon Analysis (gamma=1)')
    df['iterations'].plot(ax=ax[0], xlabel='epsilon', ylabel='iterations', logx=True, ylim=[0, 10])
    df['scores'].plot(ax=ax[1], xlabel='epsilon', ylabel='reward', logx=True, ylim=[0, 1])
    df['times'].plot(ax=ax[2], xlabel='epsilon', ylabel='seconds', logx=True)
    plt.savefig('figures/frozen_lake_policy_iteration_epsilon.png')

    gammas = [0.9, 0.95, 0.99, 0.999]
    epsilon = 1e-20
    scores = []
    times = []
    iterations = []
    for gamma in gammas:
        startTime = time.time()
        optimal_policy, it = policyIteration(env, gamma, epsilon)
        endTime = time.time()
        score = evaluatePolicy(env, optimal_policy, gamma, 1000)
        t = endTime - startTime
        scores.append(score)
        times.append(t)
        iterations.append(it)
        print(f'Best score: {score}, Iterations: {it}, Time: {endTime-startTime}')

    df = pandas.DataFrame(data={
        'gammas': gammas,
        'iterations': iterations,
        'times': times,
        'scores': scores
    }, index=gammas)
    fig, ax = plt.subplots(1, 3, figsize=(24, 5))
    fig.suptitle('Policy Iteration - Gamma Analysis (epsilon=1e-20)')
    df['iterations'].plot(ax=ax[0], xlabel='gamma', ylabel='iterations', ylim=[0, 10])
    df['scores'].plot(ax=ax[1], xlabel='gamma', ylabel='reward', ylim=[0, 1])
    df['times'].plot(ax=ax[2], xlabel='gamma', ylabel='seconds')
    plt.savefig('figures/frozen_lake_policy_iteration_gamma.png')


def q_learn(env):
    scores = []
    times = []
    iterations = []
    alphas = [0.01, 0.05, 0.1]
    for alpha in alphas:
        startTime = time.time()
        q_policy, it, _, _ = q_learning(env, alpha=alpha)
        endTime = time.time()
        score, epsilon, _, _ = test_qpolicy(env, q_policy)
        t = endTime - startTime
        scores.append(score)
        times.append(t)
        iterations.append(it)
        print(f'Best score: {score}, Iterations: {it}, Time: {endTime-startTime}')
    df = pandas.DataFrame(data={
        'alphas': alphas,
        'iterations': iterations,
        'times': times,
        'scores': scores
    }, index=alphas)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Q-Learning - Initial Alpha Analysis (gamma=0.9)')
    df['scores'].plot(ax=ax[0], xlabel='initial alpha', ylabel='reward', ylim=[0, 1])
    df['times'].plot(ax=ax[1], xlabel='initial alpha', ylabel='seconds')
    plt.savefig('figures/frozen_lake_qlearning_alpha.png')

    scores = []
    times = []
    iterations = []
    gammas = [0.9, 0.95, 0.99, 0.999]
    for gamma in gammas:
        startTime = time.time()
        q_policy, it, _, _ = q_learning(env, discount=gamma)
        endTime = time.time()
        score, epsilon, _, _ = test_qpolicy(env, q_policy)
        t = endTime - startTime
        scores.append(score)
        times.append(t)
        iterations.append(it)
        print(f'Best score: {score}, Iterations: {it}, Time: {endTime-startTime}')
    df = pandas.DataFrame(data={
        'gammas': gammas,
        'iterations': iterations,
        'times': times,
        'scores': scores
    }, index=gammas)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Q-Learning - Gamma Analysis (alpha=0.1)')
    df['scores'].plot(ax=ax[0], xlabel='gamma', ylabel='reward', ylim=[0, 1])
    df['times'].plot(ax=ax[1], xlabel='gamma', ylabel='seconds')
    plt.savefig('figures/frozen_lake_qlearning_gamma.png')

value_iteration(env)
policy_iteration(env)
q_learn(env)