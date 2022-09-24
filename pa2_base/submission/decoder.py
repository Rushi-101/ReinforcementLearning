import argparse, time
from os import environ
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--value-policy")
parser.add_argument("--states")
parser.add_argument("--player-id")
args = parser.parse_args()

value_policy = args.value_policy
states = args.states
player_identity = args.player_id
print(player_identity)
with open(states) as f:
    states = f.readlines()
with open(value_policy) as f:
    value_policy = f.readlines()
a = []
for line in value_policy[3:]:
    a.append(line.split())
for i in range(len(states)):
    one_hot_encode = [0]*9
    if float(a[i][0]!= 0):
        one_hot_encode[int(a[i][1])] = 1
    else:
        one_hot_encode[states[i][:-1].index('0')] = 1
    print(states[i][:-1] + ' ' + str(one_hot_encode[0]) + ' ' + str(one_hot_encode[1]) + ' ' + str(one_hot_encode[2])  + ' ' + str(one_hot_encode[3]) + ' '  + str(one_hot_encode[4]) + ' '  + str(one_hot_encode[5]) + ' '  + str(one_hot_encode[6]) + ' '   + str(one_hot_encode[7]) + ' '  + str(one_hot_encode[8]))
