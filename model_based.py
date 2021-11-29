"""Brute force"""

import numpy 
import time 
import gym

"""
		Args:
		poicy [S,A] shaped matrix representing policy.
		env. OpenAi gym env.v.
			env.P represents the transition propablities of the env
			env.P[s][a] is a list of transition tuples 
			env.nS = is a number of states
			env.nA is a number of actions
		gamma: discount factor
		render: boolean to turn rendering on/off 
"""

def execute(env, policy, gamma=1.0, render=False):
	start = env.reset()
	totalReward = 0
	stepIndex = 0
	while True:
		if render:
			env.render()
		start, reward, done,_ = env.step(int(policy[start]))
		totalReward += (gamma ** stepIndex * reward)
		stepIndex += 1
		if done:
			break
	return totalReward
		
#Evaluation
def evaluatePolicy(env, policy, gamma=1.0, n=100):
	scores = [execute(env, policy, gamma, False) for _ in range(n)]
	return numpy.mean(scores)
	
#choosing a policy given a value-function
def calculatePolicy(env, v, gamma=1.0):
	policy = numpy.zeros(env.env.nS)
	for s in range(env.env.nS):
		q_sa = numpy.zeros(env.action_space.n)
		for a in range(env.action_space.n):
			for next_sr in env.env.P[s][a]:
				p, s_, r, _ = next_sr
				q_sa[a] += (p * (r + gamma * v[s_]))
		policy[s] = numpy.argmax(q_sa)
	return policy
	
#Value Iteration Algorithm
def valueIteration(env, gamma=1.0, eps=1e-20):
	statistics = []
	value = numpy.zeros(env.env.nS)
	max_iterations = 10000
	for i in range(max_iterations):
		prev_v = numpy.copy(value)
		for s in range(env.env.nS):
			q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)]
			value[s] = max(q_sa)
		delta = numpy.sum(numpy.fabs(prev_v - value))
		if (delta <= eps):
			break
		if i % 500 == 0:
			statistics.append((i, numpy.copy(value), delta))
	statistics.append((i, numpy.copy(value), delta))
	return value, i, statistics

#Iteratively calculates value-function under policy   
def CalcPolicyValue(env, policy, gamma=1.0, eps=0.1):
	value = numpy.zeros(env.env.nS)
	while True:
		previousValue = numpy.copy(value)
		for states in range(env.env.nS):
			policy_a = policy[states]
			value[states] = sum([p * (r + gamma * previousValue[s_]) for p,s_, r, _ in env.env.P[states][policy_a]])
		if (numpy.sum((numpy.fabs(previousValue - value))) <= eps):
			break
			print( "breaked")
	return value
	
	
#Policy Iteration algorithm
def policyIteration(env, gamma=1.0, eps=0.1):
	policy = numpy.random.choice(env.env.nA, size=(env.env.nS))
	maxIterations = 100000
	gamma = 1.0
	for i in range(maxIterations):
		oldPolicyValue = CalcPolicyValue(env, policy, gamma, eps)
		newPolicy = calculatePolicy(env, oldPolicyValue, gamma)
		if (numpy.all(policy == newPolicy)):
			print('Policy Iteration converged at %d' %(i+1))
			break
		policy = newPolicy
	return policy, i+1


def q_learning(env, discount=0.9, total_episodes=1e5, alpha=0.1, decay_rate=None,
               min_epsilon=0.01):
    
    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n
    
    qtable = numpy.zeros((number_of_states, number_of_actions))
    learning_rate = alpha
    gamma = discount

    # exploration parameter
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    
    if not decay_rate:
        decay_rate = 1./total_episodes
    
    rewards = []
    for episode in range(int(total_episodes)):
        # reset the environment
        state = env.reset()
        step = 0
        done = False
        total_reward = 0
        while True:

            # choose an action a in the corrent world state
            exp_exp_tradeoff = numpy.random.uniform(0,1)

            # if greater than epsilon --> exploit
            if exp_exp_tradeoff > epsilon:
                b = qtable[state, :]
                action = numpy.random.choice(numpy.where(b == b.max())[0])
#                 action = np.argmax(qtable[state, :])
            # else choose exploration
            else:
                action = env.action_space.sample()

            # take action (a) and observe the outcome state (s') and reward (r)    
            new_state, reward, done, info = env.step(action)
            total_reward += reward
            # update Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max(Q (s', a') - Q(s,a))]
            if not done:
                qtable[state, action] = qtable[state, action] + learning_rate*(reward + gamma*numpy.max(qtable[new_state, :]) - qtable[state, action])
            else:
                qtable[state, action] = qtable[state,action] + learning_rate*(reward - qtable[state,action])

            # change state
            state = new_state

            # is it Done
            if done:
                break
                
        # reduce epsilon 
        rewards.append(total_reward)
        epsilon = max(max_epsilon -  decay_rate * episode, min_epsilon) 
    #     print (epsilon)
    
    print("Solved in: {} episodes".format(total_episodes))
    return numpy.argmax(qtable, axis=1), total_episodes, qtable, rewards

def test_qpolicy(env, policy, n_epoch=1000):
    rewards = []
    episode_counts = []
    for i in range(n_epoch):
        current_state = env.reset()
        ep = 0
        done = False
        episode_reward = 0
        while not done and ep < 10000:
            ep += 1
            act = int(policy[current_state])
            new_state, reward, done, _ = env.step(act)
            episode_reward += reward
            current_state = new_state
        rewards.append(episode_reward)
        episode_counts.append(ep)
    
    # all done
    mean_reward = sum(rewards)/len(rewards)
    mean_eps = sum(episode_counts)/len(episode_counts)
    return mean_reward, mean_eps, rewards, episode_counts