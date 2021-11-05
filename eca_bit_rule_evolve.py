import gym
import random
import time
import matplotlib.pyplot as plt
from bit_rule import Bit_rule


''' evolutionary constants '''
POPULATION_SIZE = 10            # number of rules stored
NUM_PARENTS = 10                # number of rules to keep for next generation (only used in the crossover function)
MUTATION_RATE = 0.05            # chance for a bit to flip

''' rule constants '''
INPUT_SIZE = 2                  # length of bit-input offset (1 for 3 bit inputs, 2 for 5 bit inputs)

''' env constants '''
SHOW = False
NUM_OF_RULES_TO_FIND = 1        # for finding multiple rules
EPOCHS = 50                     # training rounds
NUM_TRIES = 1                   # tries each rule has each epoch
GOAL_STEPS = 10_000             # forced stop after this many steps




########## Evolutionary Algorithms ##########

def crossover(sorted_population):
    '''
    uses the fittest in the population to create the next generation
    with the uniform crossover algorithm.
    Parrent A and B can be the same
    '''
    children = []
    next_gen = []
    for idx in range(NUM_PARENTS):
        # picks the parrents
        next_gen.append(sorted_population[idx])

    for i in range(NUM_PARENTS):
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

def single_parrent(sorted_population):
    ''' picks the fittest parrent, every child is the same as the fittest parrent '''
    children = []
    next_gen = []

    parrent = sorted_population[0]
    next_gen.append(parrent)

    for _ in range(POPULATION_SIZE-1):
        child = []
        for i in range(len(parrent.rule_arry)):
            child.append(parrent.rule_arry[i])
        children.append(child)

    return children, next_gen

def mutate(child):
    ''' every element has the chance of mutation_rate to flip '''
    for i in range(len(child)):
        if (random.random() < MUTATION_RATE):
            if (child[i] == 0):
                child[i] = 1
            else: 
                child[i] = 0
    return child

def next_generation(population):
    '''
    3 step function:
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
        next_gen.append(Bit_rule(child))
    
    if (len(next_gen) < POPULATION_SIZE):
        ''' if the next population is less than the population size, fill the population with new random rules '''
        for _ in range(POPULATION_SIZE-len(next_gen)):
            rule_arry = [random.randint(0, 1) for _ in range(rule_length)]
            next_gen.append(Bit_rule(rule_arry))

    return next_gen




########## random rule initialization ##########

if INPUT_SIZE < 2:
    rule_length = 8
else:
    rule_length = 32
population = []
for rule in range(POPULATION_SIZE):
    rule_arry = [random.randint(0, 1) for _ in range(rule_length)]

    ''' rule_arry can be used to test a rule, or continue evolving from the given rule '''
    # rule_arry = 
    population.append(Bit_rule(rule_arry))



########## Training ##########

epoch_stats = {'epoch': [], 'avg': [], 'max': [], 'min': []}
rule_stats = {'rule': [], 'fitnes': []}

start = time.time()

env = gym.make('CartPole-v0').env
for _ in range(NUM_OF_RULES_TO_FIND):

    for epoch in range(1, EPOCHS+1):
        mean_scores = []

        for rule in population:
            scores = []
            steps = []
            
            for t in range(NUM_TRIES):
                score = 0
                step = 0
                observation = env.reset()

                for _ in range(GOAL_STEPS):
                    if SHOW:
                        env.render()
                    
                    action = rule.get_action(observation)
                    
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

            rule.fitnes = sum(score for score in scores)/NUM_TRIES
            avg_steps = sum(step for step in steps)/NUM_TRIES
            rule_stats['rule'].append(rule.bitlist_to_int())
            rule_stats['fitnes'].append(avg_steps)
            if steps == GOAL_STEPS:
                break

        sorted_population = sorted(population, key = lambda rule: rule.fitnes, reverse=True)
        epoch_stats['epoch'].append(epoch)
        epoch_stats['avg'].append(sum(rule.fitnes for rule in population)/POPULATION_SIZE)
        epoch_stats['max'].append(sorted_population[0].fitnes)
        epoch_stats['min'].append(sorted_population[-1].fitnes)

        print("epoch: {} avg: {} max: {} min: {}".format(epoch, round(epoch_stats['avg'][-1], 1), epoch_stats['max'][-1], epoch_stats['min'][-1]))
        print(sorted_population[0])

        if sorted_population[0].fitnes == GOAL_STEPS:
            print(f'found a rule that reached {GOAL_STEPS} steps after {epoch} generations in {time.time() - start} secunds')
            start = time.time()
            population = []
            for rule in range(POPULATION_SIZE):
                rule_arry = [random.randint(0, 1) for _ in range(rule_length)]
                population.append(Bit_rule(rule_arry))
            break

        ''' evolv and mutate population '''
        if epoch < EPOCHS:
            population = next_generation(population)

        # print()



env.close()

end = time.time()
print("took: ",end - start, "seconds")

print("-- Population --")
for rule in population:
    print(rule)

plt.plot(epoch_stats['epoch'], epoch_stats['avg'], label='avg')
plt.plot(epoch_stats['epoch'], epoch_stats['max'], label='max')
plt.plot(epoch_stats['epoch'], epoch_stats['min'], label='min')
plt.legend(loc=4)
plt.show()

plt.plot(rule_stats['rule'], rule_stats['fitnes'], 'ro', ms=1)
plt.show()