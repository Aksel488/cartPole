import gym
import time
import matplotlib.pyplot as plt
from bit_rule import Bit_rule

'''
File for testing a spesific rule
'''


''' env constants '''
SHOW = False
num_tries = 100               # tries each rule has each epoch
goal_steps = 10_000           # forced stop after this many steps

''' Rule to be tested, remember to change input_size and row_width in bit_rule file '''
# rule_arry = [1, 1, 1, 0, 1, 0, 1, 0] # 8  width
# rule_arry = [1, 1, 0, 0, 1, 1, 0, 0] # 16 width
# rule_arry = [1, 1, 1, 1, 0, 0, 1, 0] # 24 width
# rule_arry = [1, 1, 0, 0, 1, 1, 1, 0] # 32 width
# rule_arry = [1, 1, 0, 1, 1, 1, 1, 0] # 64 width

# rule_arry = [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0] # 8  width
# rule_arry = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0] # 16 width
# rule_arry = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0] # 24 width
rule_arry = [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] # 32 width
# rule_arry = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0] # 64 width
rule = Bit_rule(rule_arry)

bits = ''
for i in rule.rule_arry:
    bits += str(i)
print(f'bits: {bits} decimal: {rule.bitlist_to_int()}')


rule_stats = {'try': [], 'steps': [], 'score': []}
start = time.time()

env = gym.make('CartPole-v0').env
for t in range(num_tries):
    steps = 0
    score = 0
    observation = env.reset()

    for j in range(goal_steps):
        if SHOW:
            env.render()
        
        action = rule.get_action(observation)
        
        observation, reward, done, info = env.step(action)

        steps += reward
        score += reward - abs((observation[2]/(env.observation_space.high[2]*0.5))**2 + 5*(observation[0]/(env.observation_space.high[0]*0.5))**2)

        if (steps % 100_000 == 0):
            print(f'steps: {steps} score: {score}')

        if done:
            break

    if (t % 1 == 0):
        print(f'try: {t+1} steps: {steps} score: {score}')

    rule_stats['try'].append(t)
    rule_stats['steps'].append(steps)
    rule_stats['score'].append(score)

env.close()

end = time.time()
print("took: ",end - start, "seconds")

sum_steps = 0
sum_score = 0
for i in range(num_tries):
    sum_steps += rule_stats['steps'][i]
    sum_score += rule_stats['score'][i]
print(f'avg steps: {sum_steps/num_tries} avg fitness: {sum_score/num_tries}')

plt.plot(rule_stats['try'], rule_stats['steps'])
plt.plot(rule_stats['try'], rule_stats['score'])
plt.show()