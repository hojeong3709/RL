"""
Taxi MAP ( 5 X 5 )
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",

Description:
There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue).
When the episode starts, the taxi starts off at a random square and the passenger is at a random location.
The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination
(another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off,
the episode ends.

Observations:
There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger
(including the case when the passenger is in the taxi), and 4 destination locations.
"""
import gym
import numpy as np

if __name__ == "__main__":
    env = gym.make('Taxi-v3')
    q_value_table = np.zeros([env.observation_space.n, env.action_space.n])

    lr = 0.01
    discount_factor = 0.95
    num_episodes = 20000

    reward_list = []

    for i in range(num_episodes):
        done = False
        state_change_cnt = 0
        total_reward_sum = 0
        current_state = env.reset()
        while True:
            env.render()
            current_action = np.argmax(q_value_table[current_state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
            next_state, reward, done, _ = env.step(current_action)
            q_value_table[current_state, current_action] = q_value_table[current_state, current_action] + \
                                                           lr * (reward + discount_factor * np.max(q_value_table[next_state, :]) -
                                                           q_value_table[current_state, current_action])
            total_reward_sum += reward
            current_state = next_state
            state_change_cnt = state_change_cnt + 1

            if done:
                break

        reward_list.append((state_change_cnt, total_reward_sum))

    for episode in range(num_episodes):
        print("episode: {} state_change_count : {} total_reward : {}".format(episode, state_change_cnt, total_reward_sum))

    print(q_value_table)
