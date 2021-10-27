import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt

'''
File for testing a spesific rule
'''

_3bit_input_patterns = [
    (1,1,1),
    (1,1,0),
    (1,0,1),
    (1,0,0),
    (0,1,1),
    (0,1,0),
    (0,0,1),
    (0,0,0)
]
_5bit_input_patterns = [
    (1,1,1,1,1),
    (1,1,1,1,0),
    (1,1,1,0,1),
    (1,1,1,0,0),
    (1,1,0,1,1),
    (1,1,0,1,0),
    (1,1,0,0,1),
    (1,1,0,0,0),
    (1,0,1,1,1),
    (1,0,1,1,0),
    (1,0,1,0,1),
    (1,0,1,0,0),
    (1,0,0,1,1),
    (1,0,0,1,0),
    (1,0,0,0,1),
    (1,0,0,0,0),
    (0,1,1,1,1),
    (0,1,1,1,0),
    (0,1,1,0,1),
    (0,1,1,0,0),
    (0,1,0,1,1),
    (0,1,0,1,0),
    (0,1,0,0,1),
    (0,1,0,0,0),
    (0,0,1,1,1),
    (0,0,1,1,0),
    (0,0,1,0,1),
    (0,0,1,0,0),
    (0,0,0,1,1),
    (0,0,0,1,0),
    (0,0,0,0,1),
    (0,0,0,0,0)
]

class bit_rule:
    ''' bit array of row_width length '''
    def __init__(self, rule):
        self.rule_arry = rule
        if input_size < 2:
            # mapps the 3 bit input pattarn to the rule
            self.rule = dict(zip(_3bit_input_patterns, self.rule_arry))
        else:
            # mapps the 5 bit input pattarn to the rule
            self.rule = dict(zip(_5bit_input_patterns, self.rule_arry))
        self.fitnes = 0

    def obs_to_input(self, observation):
        output = []

        for i in range(4):
            obs_norm = observation[i]/env_max[i]
            num_of_ones = 0
            for i in np.arange(-1, 1+2/precision, 2/(precision-1)):
                if obs_norm > i:
                    num_of_ones += 1

            acc = []
            for i in range(precision):
                ''' same number of oneas and zeroes '''
                if i % 2 == 0:
                    acc.append(1)
                else:
                    acc.append(0)

            if num_of_ones < precision/2:
                ''' more zeroes than ones '''
                for i in range(int(precision/2)-num_of_ones):
                    acc[i*2] = 0
    
            if num_of_ones > precision/2:
                ''' more ones than zeroes '''
                for i in range(num_of_ones-int(precision/2)):
                    acc[(i*2)+1] = 1

            for bit in acc:
                output.append(bit)
            
        # print(output)
        # time.sleep(.1)
        return output

    def iterate(self, input):
        input = np.pad(input, (input_size, input_size), 'constant', constant_values=(0,0))
        output = np.zeros_like(input)
        for i in range(input_size, input.shape[0] - input_size):
            output[i] = self.rule[tuple(input[i-input_size:i+input_size+1])] 
        return list(output[input_size:-input_size])
            
    def get_action(self, observation, iterations):
        ''' convert the observations to an array of length row_width of 0-s and 1-s '''
        output = self.obs_to_input(observation)
        
        ''' Apply the rule '''
        for i in range(iterations):
            output = self.iterate(output)
        
        ''' Convert the last output to an action '''
        print(output, output.count(1))
        if output.count(1) > len(output)/2:
            return 1
        return 0

    def bitlist_to_int(self):
        return int("".join(str(x) for x in self.rule_arry), 2)

    def __str__(self):
        return f"{self.rule_arry} {self.fitnes}"



''' rule constants '''
input_size = 2                  # length of bit-input offset (1 for 3 bit inputs, 2 for 5 bit inputs)
row_width = 32                  # length of each row (should be divisible by 4)
precision = row_width // 4      # num of bits assigned to each observation
iterations = 1                  # number of iterations applied on each rule, each step

''' env constants '''
SHOW = True
num_tries = 1000               # tries each rule has each epoch
goal_steps = 10000             # forced stop after this many steps
env_max = [0.5, 2, 0.1, 0.5]   # max precision from the env-observation

''' Rule to be tested '''
rule_arry = [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
rule = bit_rule(rule_arry)



rule_stats = {'try': [], 'steps': []}
start = time.time()

env = gym.make('CartPole-v0').env
for t in range(num_tries):
    steps = 0
    observation = env.reset()

    for j in range(goal_steps):
        if SHOW:
            env.render()
        
        action = rule.get_action(observation, iterations)
        
        observation, reward, done, info = env.step(action)
        
        steps += reward

        if done:
            break

    if (t % 10 == 0):
        print(f'try: {t+1} steps: {steps}')

    rule_stats['try'].append(t)
    rule_stats['steps'].append(steps)

env.close()

end = time.time()
print("took: ",end - start, "seconds")


plt.plot(rule_stats['try'], rule_stats['steps'])
plt.show()