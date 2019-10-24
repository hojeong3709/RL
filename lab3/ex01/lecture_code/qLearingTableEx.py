import gym
import numpy as np

env = gym.make('FrozenLake-v0')
Q_table = np.zeros([env.observation_space.n, env.action_space.n])
print(env.render())
lr = 0.8
g = 0.95
num_episodes = 2000

rList = []

for i in range(num_episodes):
    current_state = env.reset()
    rewardAll = 0
    done = False
    j = 0
    while j < 99:
        j+=1
        action = np.argmax(Q_table[current_state,:] +
                           np.random.randn(1, env.action_space.n)*(1./(i+1)))
        next_state, reward, done, _ = env.step(action)
        Q_table[current_state, action] = Q_table[current_state, action] + \
            lr * (reward + g*np.max(Q_table[next_state, :]) - Q_table[current_state, action])
        rewardAll += reward
        current_state = next_state
        if done == True:
            break
    rList.append(rewardAll)
print('score over time:'+ str(sum(rList)/num_episodes))
print('\nq table\n', Q_table)
