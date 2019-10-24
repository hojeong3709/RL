import gym
import numpy as np

print(gym.__file__)
print(dir(gym))

env = gym.make('FrozenLake-v0')
q_table = np.zeros([env.observation_space.n, env.action_space.n])
print(env.render())

if __name__ == "__main__":

    lr = 0.8
    discount_factor = 0.95
    num_episodes = 2000

    reward_list = []
    current_state = env.reset()

    for i in range(num_episodes):
        reward_total_sum = 0
        done = False
        j = 0
        while j < 99:
            j += 1
            action = np.argmax(q_table[current_state, :] + np.random.randn(1, env.action_space.n) * (1./(i + 1)))
            next_state, reward, done, _ = env.step(action)
            q_table[current_state, action] = q_table[current_state, action] + \
                                             lr * (reward + discount_factor * np.max(q_table[next_state, :]) -
                                             q_table[current_state, action])
            reward_total_sum += reward
            current_state = next_state
            if done:
                break

        reward_list.append(reward_total_sum)

    print('score over time:' + str(sum(reward_list)/num_episodes))
    print(q_table)