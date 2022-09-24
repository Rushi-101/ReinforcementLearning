import pulp, argparse, time
import numpy as np
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--mdp")
parser.add_argument("--algorithm")
args = parser.parse_args()

mdp = args.mdp
algorithm = args.algorithm
with open(mdp) as f:
    content = f.readlines()
name = [content[i].split() for i in range(len(content))]

numStates = int(name[0][1])
numActions = int(name[1][1])
transit = [[[] for a in range(numActions)] for s in range(numStates)]
for line in name:
    if line[0] == 'transition':
        i, j = int(line[1]), int(line[2])
        transit[i][j].append((int(line[3]), float(line[4]), float(line[5])))
    elif line[0] == 'mdptype':
        type = line[1]
    elif line[0] == 'end':
        end = [int(i) for i in line[1:]]
    elif line[0] == 'discount':
       discount = float(line[1])

transition = transit
tolerance  = 1e-12

def ValueEvaluation(policy, numStates, numActions, tolerance, transition, discount):
    V0 = np.ones(numStates)
    V1 = np.zeros(numStates)
    while(np.linalg.norm(V0-V1) > tolerance):
        V0 = deepcopy(V1)
        for state in range(numStates):
            option    = transition[state][int(policy[state])]
            V1[state] = 0
            for prob in option:
                V1[state] += prob[2]*(prob[1] + discount*V0[prob[0]])
    return [round(state,6) for state in V1]

def actvalEvaluation(V, state, action, transition, discount):
    Q = 0
    option = transition[state][action]
    for s in option:
        Q += s[2]*(s[1] + discount*V[s[0]])
    return round(Q,6)

def valueIteration(numStates, numActions, transition, discount, tolerance):    
    V0 = np.ones(numStates)
    V1 = np.zeros(numStates)
    pi = np.zeros(numStates)
    t  = 0

    while(np.linalg.norm(V0-V1) > tolerance):
        V0 = deepcopy(V1)
        for state in range(numStates):
            action = 0
            values = []
            for action in range(numActions):
                option = transition[state][action]
                value  = 0
                for T in option:
                    value += T[2]*(T[1] + discount*V0[T[0]])
                values.append(value)
            V1[state]  = np.max(values)
            pi[state]  = np.argmax(values)
        t = t + 1
    return V1, pi

def linearProgramming(numStates, numActions, transition, discount, tolerance):
    problem = pulp.LpProblem("ValueFn", pulp.LpMinimize)
    cost = 0   
    decision_variables = []
    for state in range(numStates):
        variable = str('V' + str(state))
        variable = pulp.LpVariable(str(variable))
        decision_variables.append(variable)
        cost += decision_variables[state]
    problem += cost

    for state in range(numStates):
        for action in range(numActions):
            problem += ( decision_variables[state] >= pulp.lpSum([s[2]*(s[1] + discount*decision_variables[s[0]]) for s in transition[state][action]]))
    problem.writeLP("LinearProgramming.lp")
    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    V0 = list(np.zeros(numStates))
    i = 0
    for v in problem.variables():
        V0[i] = round(v.varValue, 6)
        i += 1

    pi = list(np.zeros(numStates))
    for state in range(numStates):
        value  = V0[state]
        actVal = [actvalEvaluation(V0, state, action, transition, discount) for action in range(numActions)]
        action = (np.abs(np.asarray(actVal) - value)).argmin() 
        pi[state] = action
    V0 = ValueEvaluation(pi, numStates, numActions, tolerance, transition, discount)
    return V0, pi

def policyIteration(numStates, numActions, transition, discount, tolerance):    
    pi   = np.array([0 for i in range(numStates)])
    flag = True
    while flag:
        flag = False
        V  = ValueEvaluation(pi, numStates, numActions, tolerance, transition, discount)
        for state in range(numStates):
            stateVal = V[state]
            actionIA = -1
            actValIA = stateVal
            for action in range(numActions):
                actVal = actvalEvaluation(V, state, action, transition, discount)
                if abs(actVal - stateVal) < 3e-6:
                    continue
                if actVal > actValIA:
                    actionIA = action
                    actValIA = actVal
            if actionIA != -1:
                flag = True
                pi[state] = actionIA
    return V, pi

if (algorithm == 'hpi'):
    V0, pi = policyIteration(numStates, numActions, transition, discount, 1e-9)
elif (algorithm == 'lp'):
    V0, pi = linearProgramming(numStates, numActions, transition, discount, tolerance)
elif (algorithm == 'vi'):
    V0, pi = valueIteration(numStates, numActions, transition, discount, tolerance)

for i in range(len(V0)):
    print('{:.6f}'.format(round(V0[i], 6)) + "\t" + str(int(pi[i])))