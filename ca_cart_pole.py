import gym
import time

def getAction(observation):
    if observation[2] > 0 and observation[3] > 0:
        return 1
    return 0

env = gym.make('CartPole-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        
        #action = env.action_space.sample()
        action = getAction(observation)

        observation, reward, done, info = env.step(action)
        print(action)
        time.sleep(0.5)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

