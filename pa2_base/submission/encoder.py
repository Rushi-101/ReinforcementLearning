import argparse, time
from os import environ
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--policy")
parser.add_argument("--states")
args = parser.parse_args()

policy = args.policy
states = args.states

with open(policy) as f:
    content_policy = f.readlines()

if content_policy[0][:-1]=='1':
    agent = 2
    environ = 1
else:
    agent = 1
    environ = 2

with open(states) as f:
    content_states = f.readlines()

policy = []
for i in range(len(content_policy)-1):
    policy.append(content_policy[i+1].split())

def grid_states(state):
    grid = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            grid[i][j] = state[i*3 + j]
    return grid

def end_game_cond1(grid):
    for i in range(0, 9, 3):
        if grid[i]==grid[i+1] and grid[i+1] == grid[i+2] and grid[i] != '0':
            return 1
    for j in range(3):
        if grid[j]==grid[j+3] and grid[j+3] == grid[j+6] and grid[j] != '0':
            return 1
    if grid[0]==grid[4] and grid[4] == grid[8] and grid[4] != '0':
        return 1
    if grid[2]==grid[4] and grid[4] == grid[6] and grid[4] != '0':
        return 1
    if '0' not in grid:
        return 2
    return 0

numStates = len(content_states) + 3
numActions = 9
end_states = [0,1,2]
discount = 1

f = open("mdpfile.txt", "w")
f.write('numStates' + ' ' + str(numStates) + '\n')
f.write('numActions' + ' ' + str(numActions) + '\n')
f.write('end 2 0 1' + '\n')

dict = {}
key = 3
for i in range(numStates):
    content_states[i] = content_states[i][:-1]

    if content_states[i] not in dict:
        dict[content_states[i]] = key
        key+=1
    for k in range(len(content_states[i])):
        if content_states[i][k]=='0':
            intermediate_state = content_states[i][:k] + str(agent) + content_states[i][k+1:]
            #a = grid_states(intermediate_state)
            b = end_game_cond1(intermediate_state)
            if b==1:
                reward = 0
                transition = 'transition' + ' ' + str(dict[content_states[i]]) + ' ' + str(k) + ' ' + str(0) + ' ' + str(reward) + ' ' + str(1)
                print(transition)
                f.write(transition + '\n')
                continue
            elif b==2:
                transition = 'transition' + ' ' + str(dict[content_states[i]]) + ' ' + str(k) + ' ' + str(1) + ' ' + str(0) + ' ' + str(1)
                print(transition)
                f.write(transition + '\n')
                continue                
            prob_w = 0
            win = 0
            for j in range(len(content_policy)-1):
                if policy[j][0] == intermediate_state:
                    index = j
                    break
            for l in range(numActions):
                if float(policy[index][l+1]) != 0:
                    next_state = policy[index][0][:l] + str(environ) + policy[index][0][l+1:]
                    reward = 0
                    d = end_game_cond1(next_state)
                    if d==1:
                        win = 1
                        reward = 1
                        prob_w += float(policy[index][l+1])
                    elif d==2:
                        transition = 'transition' + ' ' + str(dict[content_states[i]]) + ' ' + str(k) + ' ' + str(1) + ' ' + str(0) + ' ' + str(1)
                        print(transition)
                        f.write(transition + '\n')
                        continue         
                    if reward == 0:
                        if next_state not in dict:
                            dict[next_state] = key
                            key+=1
                        transition = 'transition' + ' ' + str(dict[content_states[i]]) + ' ' + str(k) + ' ' + str(dict[next_state]) + ' ' + str(reward) + ' ' + policy[index][l+1]
                        print(transition)
                        f.write(transition + '\n')
            if win==1:
                transition = 'transition' + ' ' + str(dict[content_states[i]]) + ' ' + str(k) + ' ' + str(2) + ' ' + str(1) + ' ' + str(prob_w)
                print(transition)
                f.write(transition + '\n')

f.write('mdptype episodic' + '\n')
f.write('discount 1' + '\n')
f.close()
#with open('readme.txt', 'w') as f:
  #  f.writelines(lines)