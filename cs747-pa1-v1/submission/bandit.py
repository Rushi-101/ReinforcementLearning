from random import random
import argparse
import numpy as np
import matplotlib
import os

parser = argparse.ArgumentParser()
parser.add_argument('--instance', type=str)
parser.add_argument('--randomSeed', type=int)
parser.add_argument('--horizon', type=int)
parser.add_argument('--epsilon', type=float)
parser.add_argument('--threshold', type=float)
parser.add_argument('--scale', type=float)
parser.add_argument('--algorithm', type=str)
args = parser.parse_args()
arms = list()
randomSeed = args.randomSeed
inst = args.instance
hz =  args.horizon
epsilon = args.epsilon
c = args.scale
th = args.threshold
algorithm = args.algorithm

np.random.seed(randomSeed)

f = open(inst, "r")
list=[]
for lines in f:
  list.append(lines.rstrip('\n').split(" "))
list = list(np.float_(list))


def epsilon_greedy_t1(instance,epsilon,hz):
    empirical_mean = np.zeros((len(instance), 1))
    pulls = np.ones((len(instance),1))
    rewards = np.zeros((len(instance),1))
    for i in range(hz):

        a = np.random.choice(np.array([0, 1]), p=[epsilon, 1-epsilon])
        if a==0:
            arm = np.random.randint(0, len(instance))
            pulls[arm] += 1
            reward = np.random.choice(np.array([0, 1]), p=[1 - instance[arm][0], instance[arm][0]])
            rewards[arm] += reward
            empirical_mean = rewards/pulls

        else:
            arm = np.argmax(empirical_mean)
            pulls[arm] += 1
            reward = np.random.choice(np.array([0, 1]), p=[1 - instance[arm][0], instance[arm][0]])
            rewards[arm] += reward
            empirical_mean = rewards/pulls
    pulls = pulls-1
    p_star = instance[np.argmax(instance)]
    max_reward = hz*p_star
    REW = np.sum(rewards)
    REG = max_reward - REW
    HIGHS = 0
    return REG[0]

def ucb_t1(instance,hz):
    empirical_mean = np.zeros((len(instance), 1))
    pulls = np.ones((len(instance),1))
    rewards = np.zeros((len(instance),1))
    ucb = np.zeros((len(instance),1))
    for i in range(hz):
        arm = np.argmax(ucb)
        pulls[arm] += 1
        reward = np.random.choice(np.array([0, 1]), p=[1 - instance[arm][0], instance[arm][0]])
        rewards[arm] += reward
        empirical_mean = rewards/pulls
        ucb = empirical_mean + np.sqrt(2*np.log(i+1)/pulls)

    p_star = instance[np.argmax(instance)]
    max_reward = hz*p_star
    REW = np.sum(rewards)
    REG = max_reward - REW
    HIGHS = 0

    return REG[0]

def KL(p, q):
    if p == 1:
        return p*np.log(p/q)
    elif p == 0:
        return (1-p)*np.log((1-p)/(1-q))
    else:
        return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))


def solve_q(rhs, p_a):
    if p_a == 1:
        return 1
    q = (p_a + 1)/2
    while KL(p_a, q) <= rhs:
        q += 0.01
    while KL(p_a, q) > rhs:
        q -= 0.01
    return q



def kl_ucb_t1(instance,c,hz):
    empirical_mean = np.zeros((len(instance), 1))
    pulls = np.ones((len(instance),1)) + 1
    rewards = np.zeros((len(instance),1))
    ucb_kl = np.zeros((len(instance),1))
    for i in range(hz):
        for j in range(len(instance)):
            rhs = (np.log(i+2) + c*np.log(np.log(i+2)))/pulls[j]
            ucb_kl[j] = solve_q(rhs, empirical_mean[j])

        arm = np.argmax(ucb_kl)
        pulls[arm] += 1
        reward = np.random.choice(np.array([0, 1]), p=[1 - instance[arm][0], instance[arm][0]])
        rewards[arm] += reward
        empirical_mean = rewards/pulls

    p_star = instance[np.argmax(instance)]
    max_reward = hz*p_star
    REW = np.sum(rewards)
    REG = max_reward - REW
    HIGHS = 0
    return REG[0]

def thompson_sampling_t1(instance,hz):

    empirical_mean = np.zeros((len(instance), 1))
    pulls = np.ones((len(instance),1))
    rewards = np.zeros((len(instance),1))
    beta = np.zeros((len(instance),1))
    for i in range(hz):
        for j in range(len(instance)):
            beta[j] = np.random.beta(rewards[j]+1, pulls[j]-rewards[j]+1)

        arm = np.argmax(beta)
        pulls[arm] += 1
        reward = np.random.choice(np.array([0, 1]), p=[1 - instance[arm][0], instance[arm][0]])
        rewards[arm] += reward
        empirical_mean = rewards/pulls

    p_star = instance[np.argmax(instance)]
    max_reward = hz*p_star
    REW = np.sum(rewards)
    REG = max_reward - REW
    HIGHS = 0

    return REG[0]


def ucb_t2(instance,c,hz):
    empirical_mean = np.zeros((len(instance), 1))
    pulls = np.ones((len(instance),1))
    rewards = np.zeros((len(instance),1))
    ucb = np.zeros((len(instance),1))
    for i in range(hz):
        arm = np.argmax(ucb)
        pulls[arm] += 1
        reward = np.random.choice(np.array([0, 1]), p=[1 - instance[arm][0], instance[arm][0]])
        rewards[arm] += reward
        empirical_mean = rewards/pulls
        ucb = empirical_mean + np.sqrt(c*np.log(i+1)/pulls)

    p_star = instance[np.argmax(instance)]
    max_reward = hz*p_star
    REW = np.sum(rewards)
    REG = max_reward - REW
    HIGHS = 0

    return REG[0]


def alg_t3(instance,hz):
    a = np.array([instance[1],instance[2],instance[3]])
    empirical_mean = np.zeros((len(instance)-1, 1))
    pulls = np.ones((len(instance)-1,1))
    rewards = np.zeros((len(instance)-1,1))
    beta = np.zeros((len(instance)-1,1))
    for i in range(hz):
        for j in range(len(instance)-1):
            beta[j] = np.random.beta(rewards[j]+1, pulls[j]-rewards[j]+1)

        arm = np.argmax(beta)
        pulls[arm] += 1
        reward = np.random.choice(instance[0], p=[a[arm][0], a[arm][1], a[arm][2], a[arm][3], a[arm][4]])
        rewards[arm] += reward
        empirical_mean = rewards/pulls
    expectation = 0
    exp = np.zeros((len(a),1))
    for i in range(len(instance)-1):
        expectation =  instance[i+1][1]*0.25 + instance[i+1][2]*0.5 + instance[i+1][3]*0.75 + instance[i+1][4]*1
        exp[i] = expectation
        expectation = 0

    max_reward = hz*np.max(exp)  
    REW = np.sum(rewards)
    REG = max_reward - REW
    HIGHS = 0

    return REG

def alg_t4(instance,th,hz):
    a = np.array([instance[1],instance[2],instance[3]])
    empirical_mean = np.zeros((len(a), 1))
    pulls = np.ones((len(a),1))
    rewards = np.zeros((len(a),1))
    beta = np.zeros((len(a),1))
    HIGHS = 0
    for i in range(hz):
        for j in range(len(a)):
            beta[j] = np.random.beta(rewards[j]+1, pulls[j]-rewards[j]+1)

        arm = np.argmax(beta)
        pulls[arm] += 1
        reward = np.random.choice(instance[0], p=[a[arm][0], a[arm][1], a[arm][2], a[arm][3], a[arm][4]])
        if reward>th:
            HIGHS += 1
            reward = 1
        else:
            reward = 0
        rewards[arm] += reward
        empirical_mean = rewards/pulls

    expectation = 0
    exp = np.zeros((len(a),1))
    if th==0.6:
        for i in range(len(instance)-1):
            expectation =  instance[i+1][3] + instance[i+1][4]
            exp[i] = expectation
            expectation = 0
    else:
        for i in range(len(instance)-1):
            expectation =  instance[i+1][1] + instance[i+1][2] + instance[i+1][3] + instance[i+1][4]
            exp[i] = expectation
            expectation = 0
    max_reward = hz*np.max(exp)  
    REW = np.sum(rewards)
    REG = max_reward - HIGHS
    return REG,HIGHS

HIGHS = 0
REG = 0
if algorithm == "epsilon-greedy-t1":
    REG = epsilon_greedy_t1(list, epsilon, hz)
elif algorithm == "ucb-t1":
    REG = ucb_t1(list,hz)
elif algorithm == "kl-ucb-t1":
    REG = kl_ucb_t1(list,3,hz)
elif algorithm == "thompson-sampling-t1":
    REG = thompson_sampling_t1(list,hz)
elif algorithm == "ucb-t2":
    REG = ucb_t2(list,c,hz)
elif algorithm == "alg-t3":
    REG = alg_t3(list,hz)
else:
    REG,HIGHS = alg_t4(list,th,hz)

print(f'{inst}, {algorithm}, {randomSeed}, {epsilon}, {c}, {th}, {hz}, {REG}, {HIGHS}')
 