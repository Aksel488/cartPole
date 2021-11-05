import numpy as np
import matplotlib.pyplot as plt
import random

# random.seed(0)

n_rules = 100
rules_length = 1000
iterations = 5000

class BitString:
    def __init__(self, n_rules, rules_length):
        self.rules = [[random.randint(0, 1) for _ in range(rules_length)] for _ in range(n_rules)]
        self.fitnes = [0 for _ in range(n_rules)]
        for i in range(n_rules):
            self.fitnes[i] = self.getFitnes(self.rules[i])

    def getFitnes(self, row):
        ''' return number of 1-s in row '''
        return row.count(1)
    
    def splice(self):
        ''' copies a range of values from list A to B '''
        max_fitnes = max(self.fitnes)
        idx = self.fitnes.index(max_fitnes)
        A = self.rules[idx]
        B = self.rules[random.randint(0, len(self.rules) - 1)]
        start = random.randint(0, len(self.rules[0]))
        stop  = random.randint(0, len(self.rules[0]))
        if (start > stop):
            start, stop = stop, start
        B[start:stop] = A[start:stop]
        
        self.mutate(B)

    def crossover(self):
        '''
        uses half the fittest in the population to create the next generation
        with the uniform crossover algorithm.
        Parrent A and B can be the same
        '''
        children = []
        next_gen = []
        for _ in range(int(len(self.rules)/2)):
            # picks the parrents
            max_fitnes = max(self.fitnes)
            idx = self.fitnes.index(max_fitnes)
            next_gen.append(self.rules[idx])
            self.fitnes[idx] = 0 # sets the fitnes to 0 so it wont be choosen again

        for i in range(len(next_gen)):
            # creates the children
            child = []
            mask = [random.randint(0, 1) for _ in range(len(self.rules[0]))]
            parrentA = next_gen[i]
            parrentB = next_gen[random.randint(0, len(next_gen)-1)]
            for j in range(len(parrentA)):
                if (mask[j] == 1):
                    child.append(parrentA[j])
                elif (mask[j] == 0):
                    child.append(parrentB[j])
            children.append(child)
        
        for child in children:
            child = self.mutate(child)
            next_gen.append(child)

        self.rules = next_gen

    def single_parrent(self):
        children = []
        next_gen = []

        max_fitnes = max(self.fitnes)
        idx = self.fitnes.index(max_fitnes)
        parrent = self.rules[idx]
        next_gen.append(parrent)

        for _ in range(n_rules-1):
            child = []
            for i in range(len(parrent)):
                child.append(parrent[i])
            children.append(child)
        
        for child in children:
            child = self.mutate(child)
            next_gen.append(child)

        self.rules = next_gen

    def mutate(self, child):
        ''' every element has a 1% chance of flipping '''
        for i in range(len(child)):
            if (random.random() < 0.01):
                if (child[i] == 0):
                    child[i] = 1
                else: 
                    child[i] = 0
        return child


    def fit(self):
        ''' loops until one set of rules is all 1-s '''
        epochs = 0
        y_mean = []
        y_max  = []
        y_min  = []
        x      = []

        for _ in range(iterations):
            max_fitnes = max(self.fitnes)
            min_fitnes = min(self.fitnes)


            ''' different ways of evolving populations (choose only one) '''
            self.splice()
            # self.single_parrent()
            # self.crossover()

            for i in range(len(self.rules)):
                self.fitnes[i] = self.getFitnes(self.rules[i])
            
            mean_fitnes = np.mean(self.fitnes)
            print('max fitnes: {} mean fitnes: {}'.format(max_fitnes, mean_fitnes))

            ''' create plot points for mean, min and max '''
            y_mean.append(mean_fitnes)
            y_max.append(max_fitnes)
            y_min.append(min_fitnes)
            x.append(epochs)

            
            epochs += 1
            if (max(self.fitnes) == len(self.rules[0])):
                print("Finished after {} epochs".format(epochs))
                break
            
        mean_plot, = plt.plot(x, y_mean)
        max_plot, = plt.plot(x, y_max)
        min_plot, = plt.plot(x, y_min)
        plt.legend([max_plot, mean_plot, min_plot], ["max", "mean", "min"])
        plt.show()

    def show(self):
        for rule in self.rules:
            print(rule)
        print(self.fitnes)

bs = BitString(n_rules, rules_length)
bs.show()
bs.fit()
bs.show()
