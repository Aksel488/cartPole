import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt

from numpy.core.fromnumeric import mean

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

population_size = 20        # number of rules stored
rule_size = 32              # length of bit-array (8 for 3 bit inputs, 32 for 5 bit inputs)
row_width = 12              # length of each row
epochs = 20                 # training rounds
num_tries = 5               # tries each rule has each epoch
iterations = 5              # number of iterations applied on each rule, each step
goal_steps = 500            # forced stop after this many steps

class bit_rule:
    ''' bit array of rule_size length '''
    def __init__(self, rule):
        self.rule_arry = rule
        self.rule = dict(zip(_5bit_input_patterns, self.rule_arry)) # mapps the ruleset to the output
        self.fitnes = 0

    def obs_to_input(self, observation):
        output = []
        i = 0

        for j in range(row_width):
            i = j % 4
            if (observation[i] < 0):
                output.append(0)
            else:
                output.append(1)

        return output

    def iterate(self, input):
        input = np.pad(input, (2, 2), 'constant', constant_values=(0,0))
        output = np.zeros_like(input)
        for i in range(2, input.shape[0] - 2):
            output[i] = self.rule[tuple(input[i-2:i+3])] 
        return list(output[2:-2])
            
    def get_action(self, observation, iterations):
        ''' convert the observations to an array of length row_width of 0-s and 1-s '''
        output = self.obs_to_input(observation)
        
        ''' Apply the rule '''
        for i in range(iterations):
            output = self.iterate(output)
        
        ''' Convert the last output to a action '''
        if (output.count(1) > len(output) / 2):
            return 1
        return 0

    def bitlist_to_int(self):
        return int("".join(str(x) for x in self.rule_arry), 2)

    def __str__(self):
        return f"{self.rule_arry} {self.fitnes}"


def next_generation(population):
    '''
    uses half the fittest in the population to create the next generation
    with the uniform crossover algorithm.
    Parrent A and B can be the same
    '''
    children = []
    next_gen = []
    sorted_population = sorted(population, key = lambda rule: rule.fitnes, reverse=True)
    for idx in range(int(population_size/2)):
        # picks the parrents
        next_gen.append(sorted_population[idx])

    for i in range(len(next_gen)):
        # creates the children
        child = []
        mask = [random.randint(0, 1) for _ in range(len(population[0].rule_arry))]
        parrentA = next_gen[i]
        parrentB = next_gen[random.randint(0, len(next_gen)-1)]
        for j in range(len(parrentA.rule_arry)):
            if (mask[j] == 1):
                child.append(parrentA.rule_arry[j])
            elif (mask[j] == 0):
                child.append(parrentB.rule_arry[j])
        children.append(child)

        # print('parrentA:',parrentA)
        # print('parrentB:',parrentB)
        # print('mask    :',mask)
        # print('child   :',child)
    
    for child in children:
        child = mutate(child)
        next_gen.append(bit_rule(child))

    return next_gen


def mutate(child):
    ''' every element has a 5% chance of flipping '''
    for i in range(len(child)):
        if (random.random() < 0.05):
            if (child[i] == 0):
                child[i] = 1
            else: 
                child[i] = 0
    return child

env = gym.make('CartPole-v0').env

population = []
for rule in range(population_size):
    rule_arry = [random.randint(0, 1) for _ in range(rule_size)]
    # rule_arry = [0,1,0,0,1,1,1,0,0,1,1,1,1,0,1,0]
    # rule_arry = [0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,1]
    # rule_arry = [0,1,1,1,0,0,1,1,1,0,0,0,0,1,0,0]
    # rule_arry = [0,1,1,1,0,0,1,1,1,1,0,1,1,1,0,1]
    # rule_arry = [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0] ->
    # rule_arry = [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    # rule_arry = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1]
    # rule_arry = [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    # rule_arry = 
    population.append(bit_rule(rule_arry))

epoch_stats = {'epoch': [], 'avg': [], 'max': [], 'min': [],}

for epoch in range(1, epochs+1):
    mean_scores = []

    for rule in population:
        scores = []
        
        for t in range(num_tries):
            score = 0
            observation = env.reset()

            for j in range(goal_steps):
                # env.render()
                
                action = rule.get_action(observation, iterations)
                
                observation, reward, done, info = env.step(action)

                score += reward
                # if observation[0] > .1 or observation[0] < -.1:
                #     break
                if done:
                    break
            scores.append(score)
            #print(score)
        mean_score = np.mean(scores)
        rule.fitnes = mean_score
        # mean_scores.append({'Rule': rule.rule_arry, 'mean score': mean_score})
        # print(mean_scores[-1])

    # sorted_scores = sorted(mean_scores, key = lambda i: i['mean score'], reverse=True)
    sorted_population = sorted(population, key = lambda rule: rule.fitnes, reverse=True)
    epoch_stats['epoch'].append(epoch)
    epoch_stats['avg'].append(sum(rule.fitnes for rule in population)/population_size)
    epoch_stats['max'].append(sorted_population[0].fitnes)
    epoch_stats['min'].append(sorted_population[-1].fitnes)

    print("epoch: {} avg: {} max: {} min: {}".format(epoch, round(epoch_stats['avg'][-1], 1), epoch_stats['max'][-1], epoch_stats['min'][-1]))
    print(sorted_population[0])

    ''' evolv and mutate pupolation '''
    population = next_generation(population)

    print()

for rule in population:
    print(rule.fitnes," ",rule.rule_arry)
env.close()

plt.plot(epoch_stats['epoch'], epoch_stats['avg'], label='avg')
plt.plot(epoch_stats['epoch'], epoch_stats['max'], label='max')
plt.plot(epoch_stats['epoch'], epoch_stats['min'], label='min')
plt.legend(loc=4)
plt.show()