import gym
import numpy as np
import random

from numpy.core.fromnumeric import mean

eight_bit_input_patterns = [
    (1,1,1),
    (1,1,0),
    (1,0,1),
    (1,0,0),
    (0,1,1),
    (0,1,0),
    (0,0,1),
    (0,0,0)
]

class eight_bit_rule:
    ''' singe rule-set of 8 bits '''
    def __init__(self, idx):
        if idx < 256:
            outputs = list(map(int,format(idx, "#010b")[2:])) # converts a int to a array of 8-bits
            self.rule = dict(zip(eight_bit_input_patterns, outputs)) # mapps the ruleset to the output
            self.rule["name"] = "Rule %d" % (idx)
        else:
            raise ValueError("Rule is not part of the 8-bit registry")

    def iterate(self, input):
        input = np.pad(input, (1, 1), 'constant', constant_values=(0,0))
        output = np.zeros_like(input)
        for i in range(1, input.shape[0] - 1):
            output[i] = self.rule[tuple(input[i-1:i+2])] # takes value of index -1, 0, 1 of input array and applyes the rule
        return list(output[1:-1])
            
    def get_action(self, observation, iterations):
        ''' convert the observations to an array of length 8 of 0-s and 1-s '''
        input = []
        i = 0
        for j in range(8):
            if (observation[i] < 0):
                input.append(0)
            else:
                input.append(1)
            if (j % 2 == 1):
                i += 1
        
        for i in range(iterations):
            output = self.iterate(input)
            input = output
        
        if (output.count(1) > len(output) / 2):
            return 1
        return 0

    def bitlist_to_int(self, list):
        return sum([j*(2**i) for i,j in list(enumerate(reversed(list)))])


env = gym.make('CartPole-v0').env

for iter in range(1,10):
    mean_scores = []

    for n_rule in range(256):
        rule = eight_bit_rule(n_rule)
        scores = []
        
        for t in range(10):
            score = 0
            observation = env.reset()

            for _ in range(500):
                #env.render()
                
                action = rule.get_action(observation, iter)
                observation, reward, done, info = env.step(action)

                score += reward
                if done:
                    break
            scores.append(score)
        mean_scores.append({'Rule': n_rule, 'mean score': np.mean(scores)})
        #print(mean_scores[-1])

    sorted_scores = sorted(mean_scores, key = lambda i: i['mean score'])
    print("-- Top 10 rules with {} iterations --".format(iter))
    for ind in range(-1, -11, -1):
        print(sorted_scores[ind])
    print()

env.close()