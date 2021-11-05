import numpy as np
import time

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

''' rule constants '''
INPUT_SIZE = 2                  # length of bit-input offset (1 for 3 bit inputs, 2 for 5 bit inputs)
ROW_WIDTH = 32                  # length of each row (should be divisible by 4)
PRECISION = ROW_WIDTH // 4      # num of bits assigned to each observation
ITERATIONS = 1                  # number of iterations applied on each rule, each step
ENV_MAX = [0.5, 2, 0.05, 0.5]   # max precision from the env-observation

class Bit_rule:
    ''' bit array of row_width length '''
    def __init__(self, rule):
        self.rule_arry = rule
        if INPUT_SIZE < 2:
            # mapps the 3 bit input pattarn to the rule
            self.rule = dict(zip(_3bit_input_patterns, self.rule_arry))
        else:
            # mapps the 5 bit input pattarn to the rule
            self.rule = dict(zip(_5bit_input_patterns, self.rule_arry))
        self.fitnes = 0

    def obs_to_input(self, observation):
        output = []

        for i in range(4):
            obs_norm = observation[i]/ENV_MAX[i]
            num_of_ones = 0
            for i in np.arange(-1, 1+2/PRECISION, 2/(PRECISION-1)):
                if obs_norm > i:
                    num_of_ones += 1

            acc = []
            for i in range(PRECISION):
                ''' same number of ones and zeroes, eks: 10101010 '''
                if i % 2 == 0:
                    acc.append(1)
                else:
                    acc.append(0)

            if num_of_ones < PRECISION/2:
                ''' more zeroes than ones, eks: 00001010 '''
                for i in range(int(PRECISION/2)-num_of_ones):
                    acc[i*2] = 0
    
            if num_of_ones > PRECISION/2:
                ''' more ones than zeroes, eks: 11111010 '''
                for i in range(num_of_ones-int(PRECISION/2)):
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
        input = np.pad(input, (INPUT_SIZE, INPUT_SIZE), 'constant', constant_values=(0,0))
        output = np.zeros_like(input)
        for i in range(INPUT_SIZE, input.shape[0] - INPUT_SIZE):
            output[i] = self.rule[tuple(input[i-INPUT_SIZE:i+INPUT_SIZE+1])] 
        return list(output[INPUT_SIZE:-INPUT_SIZE])
            
    def get_action(self, observation):
        ''' convert the observations to an array of length row_width of 0-s and 1-s '''
        output = self.obs_to_input(observation)
        
        ''' Apply the rule '''
        for _ in range(ITERATIONS):
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
        ''' return the decimal representation of the rule_arry '''
        return int("".join(str(x) for x in self.rule_arry), 2)

    def __str__(self):
        return f"{self.rule_arry} {self.fitnes}"