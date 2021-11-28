import gym
import numpy as np

env = gym.make('FrozenLake8x8-v1')

# env.render()
# print(list(env.action_space))

# Loop
#   D <- 0
#   Loop for each s E S:
#   u <- V(s)
#   V(s) <- max_a S p(s', r|s, a)[r + g * V(s')]
#   D <- max(D, u-V(s))
# until D < theta

# output p(s) = argmax_a Sum p(s', r|s, a)[r + gV(s')]


def value_iteration(env, max_iterations=100000, gamma=0.9, epsilon=0.00001):
    V = np.zeros(env.nS)
    delta = 0
    for i in range(max_iterations):
        prev_V = V.copy()
        for state in range(env.nS):
            u = V[state]
            max_Vs = 0
            for action in range(env.nA):
                V_sum = 0
                for probability, next_state, reward, _ in env.P[state][action]:
                    V_sum += probability * (reward + gamma * prev_V[next_state])
                if V_sum >= max_Vs:
                    max_Vs = V_sum
            V[state] = max_Vs
        delta = np.sum(np.fabs(prev_V - V))
        if delta <= epsilon:
            break
    return V

def calculatePolicy(v, gamma=0.9):
  policy = np.zeros(env.env.nS)
  for s in range(env.env.nS):
    q_sa = np.zeros(env.action_space.n)
    for a in range(env.action_space.n):
      for next_sr in env.env.P[s][a]:
        p, s_, r, _ = next_sr
        q_sa[a] += (p * (r + gamma * v[s_]))
    policy[s] = np.argmax(q_sa)
  return policy

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

V = value_iteration(env)
policy = calculatePolicy(V)
execute(env, policy, render=True)
