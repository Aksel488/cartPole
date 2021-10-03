import gym
import numpy as np
import random
import time

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
rule_size = 32              # length of bit-array
epochs = 100                # training rounds
num_tries = 4               # tries each rule has each epoch
iterations = 10             # number of iterations applied on each rule, each step
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

        for j in range(rule_size):
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
        ''' convert the observations to an array of length 8 of 0-s and 1-s '''
        output = self.obs_to_input(observation)
        
        for i in range(iterations):
            tmp_output = self.iterate(output)
            output = tmp_output
        
        if (output.count(1) > len(output) / 2):
            return 1
        return 0

    def bitlist_to_int(self):
        return int("".join(str(x) for x in self.rule_arry), 2)


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
    # rule_arry = 
    population.append(bit_rule(rule_arry))

for epoch in range(epochs):
    mean_scores = []

    for rule in population:
        scores = []
        
        for t in range(num_tries):
            score = 0
            observation = env.reset()

            for j in range(goal_steps):
                #env.render()
                
                action = rule.get_action(observation, iterations)
                
                observation, reward, done, info = env.step(action)

                score += reward
                # if observation[0] > .2 or observation[0] < -.2:
                #     break
                if done:
                    break
            scores.append(score)
            #print(score)
        mean_score = np.mean(scores)
        rule.fitnes = mean_score
        mean_scores.append({'Rule': rule.rule_arry, 'mean score': mean_score})
        #print(mean_scores[-1])

    sorted_scores = sorted(mean_scores, key = lambda i: i['mean score'], reverse=True)
    print("-- Top 10 rules after {} epochs --".format(epoch+1))
    for ind in range(10):
        print(sorted_scores[ind])
    print()

    ''' evolv and mutate pupolation '''
    population = next_generation(population)

for rule in population:
    print(rule.fitnes," ",rule.rule_arry)
env.close()