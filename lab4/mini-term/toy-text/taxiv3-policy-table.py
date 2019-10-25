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
    policy_table = [[0.25, 0.25, 0.25, 0.25, 0.25, 0.25] for _ in range(env.observation_space.n)]
    value_table = np.zeros([env.observation_space.n])
    next_value_table = np.zeros([env.observation_space.n])
    discount_factor = 0.9

    num_episodes = 2000
    for i in range(num_episodes):
        current_state = env.reset()
        env.render()
        total_reward_sum = 0

        # Policy Evaluation
        for current_state in range(env.observation_space.n):
            value = [0., 0., 0., 0., 0., 0.]
            for current_action in range(env.action_space.n):
                next_state, reward, done, _ = env.step(current_action)
                if done:
                    value = 0
                    next_value_table[current_state] = value
                    continue
                total_reward_sum += reward
                next_value = value_table[next_state]
                value += policy_table[current_state][current_action] * (reward + discount_factor * next_value)

            next_value_table[current_state] = np.sum(value)
            value_table = next_value_table

        # Policy Improvement
        next_policy = policy_table
        for current_state in range(env.observation_space.n):

            value = -99999
            max_index = []

            # 반환할 정책 초기화
            result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            # 모든 행동에 대해서 [보상 + (감가율 * 다음 상태 가치함수)] 계산
            for index, current_action in enumerate(range(env.action_space.n)):
                next_state, reward, done, _ = env.step(current_action)
                if done:
                    continue

                print(index)
                print(current_action)

                next_value = value_table[next_state]
                temp = reward + discount_factor * next_value

                # 받을 보상이 최대인 행동의 index(최대가 복수라면 모두)를 추출
                if temp == value:
                    max_index.append(index)
                elif temp > value:
                    value = temp
                    max_index.clear()
                    max_index.append(index)

                # 행동의 확률 계산
                prob = 1 / len(max_index)

                for index in max_index:
                    result[index] = prob

            next_policy[current_state] = result

        policy_table = next_policy

    for episode in range(num_episodes):
        print("episode: {} total_reward : {}".format(episode, total_reward_sum))

    print(policy_table)
    print(value_table)
