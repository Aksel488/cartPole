import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt

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

''' evolutionary constants '''
population_size = 10            # number of rules stored
num_parents = 10                # number of rules to keep for next generation (only used in the crossover function)
mutation_rate = 0.05            # chance for a bit to flip

''' rule constants '''
input_size = 2                  # length of bit-input offset (1 for 3 bit inputs, 2 for 5 bit inputs)
row_width = 32                  # length of each row (should be divisible by 4)
precision = row_width // 4      # num of bits assigned to each observation
iterations = 1                  # number of iterations applied on each rule, each step

''' env constants '''
SHOW = False
epochs = 5                      # training rounds
num_tries = 10                  # tries each rule has each epoch
goal_steps = 1000               # forced stop after this many steps
env_max = [0.5, 2, 0.1, 0.5]    # max precision from the env-observation

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

        # i = 0
        # j = 1
        # for j in range(row_width):
        #     # i = 0 1 2 3 0 1 2 3 0 1 2 3 if precision=3
        #     i = j % 4
        #     if (observation[i] < 0):
        #         output.append(0)
        #     else:
        #         output.append(1)

        # for _ in range(row_width):
        #     # i = 0 0 0 1 1 1 2 2 2 3 3 3 if precision=3
        #     if (observation[i] < 0):
        #         output.append(0)
        #     else:
        #         output.append(1)
        # if j >= precision:
        #     i += 1
        #     j = 0
        # j += 1
            
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
        # # value of the center position
        # num = 0
        # num = output[int(len(output)/2)]
        # if (num > 0):
        #     return 1
        # return 0

        # value majority
        if output.count(1) > len(output)/2:
            return 1
        return 0

    def bitlist_to_int(self):
        return int("".join(str(x) for x in self.rule_arry), 2)

    def __str__(self):
        return f"{self.rule_arry} {self.fitnes}"



########## Evolutionary Algorithms ##########

def crossover(sorted_population):
    '''
    uses the fittest in the population to create the next generation
    with the uniform crossover algorithm.
    Parrent A and B can be the same
    '''
    children = []
    next_gen = []
    for idx in range(num_parents):
        # picks the parrents
        next_gen.append(sorted_population[idx])

    for i in range(num_parents):
        # creates the children
        child = []
        mask = [random.randint(0, 1) for _ in range(len(sorted_population[0].rule_arry))]
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
    return children, next_gen

def single_parrent(population):
    ''' picks the fittest parrent, every child is the same as the fittest parrent '''
    children = []
    next_gen = []

    parrent = population[0]
    next_gen.append(parrent)

    for _ in range(population_size-1):
        child = []
        for i in range(len(parrent.rule_arry)):
            child.append(parrent.rule_arry[i])
        children.append(child)

    return children, next_gen

def mutate(child):
    ''' every element has the chance of mutation_rate to flip '''
    for i in range(len(child)):
        if (random.random() < mutation_rate):
            if (child[i] == 0):
                child[i] = 1
            else: 
                child[i] = 0
    return child

def next_generation(population):
    '''
    3 sep function:
    1. pick the parrent(s) and create the children (parrents is added to next_gen)
    2. mutate the children
    3. return the next generation
    '''
    children = []
    next_gen = []
    sorted_population = sorted(population, key = lambda rule: rule.fitnes, reverse=True)
    
    ''' different algorithms for picking parrents and creating children (use only one!) '''
    # children, next_gen = crossover(sorted_population)
    children, next_gen = single_parrent(sorted_population)
    
    for child in children:
        ''' mutate the rules in the children list and add them to the next generation '''
        child = mutate(child)
        next_gen.append(bit_rule(child))
    
    if (len(next_gen) < population_size):
        ''' if the next population is less than the population size, fill the population with new random rules '''
        for _ in range(population_size-len(next_gen)):
            rule_arry = [random.randint(0, 1) for _ in range(rule_length)]
            next_gen.append(bit_rule(rule_arry))

    
    return next_gen




########## random rule initialization ##########

if input_size < 2:
    rule_length = 8
else:
    rule_length = 32
population = []
for rule in range(population_size):
    rule_arry = [random.randint(0, 1) for _ in range(rule_length)]

    ''' rule_arry can be used to test a rule, or continue evolving from the given rule '''
    # rule_arry = [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    # rule_arry = [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # rule_arry = 
    population.append(bit_rule(rule_arry))



########## Training ##########

epoch_stats = {'epoch': [], 'avg': [], 'max': [], 'min': []}
rule_stats = {'rule': [], 'fitnes': []}

start = time.time()

env = gym.make('CartPole-v0').env
for epoch in range(1, epochs+1):
    mean_scores = []

    for rule in population:
        scores = []
        steps = []
        
        for t in range(num_tries):
            score = 0
            step = 0
            observation = env.reset()

            for j in range(goal_steps):
                if SHOW:
                    env.render()
                
                action = rule.get_action(observation, iterations)
                
                observation, reward, done, info = env.step(action)
                
                step += reward
                score += reward
                # score += reward - abs((observation[2]/(env.observation_space.high[2]*0.5))**2 + 5*(observation[0]/(env.observation_space.high[0]*0.5))**2)

                if done:
                    break

            scores.append(score)
            steps.append(step)
            if SHOW:
                # print(score)
                print(step)

        rule.fitnes = sum(score for score in scores)/num_tries
        avg_steps = sum(step for step in steps)/num_tries
        rule_stats['rule'].append(rule.bitlist_to_int())
        rule_stats['fitnes'].append(avg_steps)

    sorted_population = sorted(population, key = lambda rule: rule.fitnes, reverse=True)
    epoch_stats['epoch'].append(epoch)
    epoch_stats['avg'].append(sum(rule.fitnes for rule in population)/population_size)
    epoch_stats['max'].append(sorted_population[0].fitnes)
    epoch_stats['min'].append(sorted_population[-1].fitnes)

    print("epoch: {} avg: {} max: {} min: {}".format(epoch, round(epoch_stats['avg'][-1], 1), epoch_stats['max'][-1], epoch_stats['min'][-1]))
    print(sorted_population[0])

    ''' evolv and mutate population '''
    if epoch < epochs:
        population = next_generation(population)

    print()

env.close()

print("-- Population --")
for rule in population:
    print(rule)

end = time.time()
print("took: ",end - start, "seconds")

plt.plot(epoch_stats['epoch'], epoch_stats['avg'], label='avg')
plt.plot(epoch_stats['epoch'], epoch_stats['max'], label='max')
plt.plot(epoch_stats['epoch'], epoch_stats['min'], label='min')
plt.legend(loc=4)
plt.show()

plt.plot(rule_stats['rule'], rule_stats['fitnes'], 'ro', ms=1)
plt.show()