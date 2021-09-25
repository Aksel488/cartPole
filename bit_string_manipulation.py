import numpy as np
import random

# random.seed(0)

class BitString:
    def __init__(self, n_rules, rules_length):
        self.rules = [[random.randint(0, 1) for _ in range(rules_length)] for _ in range(n_rules)]
        self.fitnes = [0 for _ in range(n_rules)]
        for i in range(n_rules):
            self.fitnes[i] = self.getFitnes(self.rules[i])

    def getFitnes(self, row):
        ''' return number of 1-s in row '''
        return row.count(1)
    
    def spalt(self, A, B, start, stop):
        ''' copies a range of values from list A to B '''
        B[start:stop] = A[start:stop]

    def mutate(self):
        ''' every element has a 2% chance of flipping '''
        for rule in self.rules:
            for i in range(len(rule)):
                if (random.random() < 0.02):
                    if (rule[i] == 0):
                        rule[i] = 1
                    else: 
                        rule[i] = 0


    def fit(self):
        ''' loops until one set of rules is all 1-s '''
        epochs = 0
        while True:
            idx = self.fitnes.index(max(self.fitnes))
            A = self.rules[idx]
            B = self.rules[random.randint(0, len(self.rules) - 1)]
            start = random.randint(0, len(self.rules[0]))
            stop  = random.randint(0, len(self.rules[0]))
            if (start > stop):
                start, stop = stop, start

            self.spalt(A, B, start, stop)
            self.mutate()

            for i in range(len(self.rules)):
                self.fitnes[i] = self.getFitnes(self.rules[i])
            
            if (epochs % 1000 == 0):
                print(sum(self.fitnes))
            epochs += 1
            if (max(self.fitnes) == len(self.rules[0])):
                print("Finished after {} epochs".format(epochs))
                break



    def show(self):
        for rule in self.rules:
            print(rule)
        print(self.fitnes)
            # print(self.getFitnes(rule))

bs = BitString(20, 30)
bs.show()
bs.fit()
bs.show()
