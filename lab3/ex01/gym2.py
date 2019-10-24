import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
tf.set_random_seed(777)

if __name__ == "__main__":
    input1 = tf.placeholder(dtype=tf.float32, shape=[1, 16])
    w = tf.Variable(tf.random_normal([16, 4], stddev=0.01))
    # Sixteen State --> Next Four Action ==> Q value
    currentQ = tf.matmul(input1, w)
    # Next Action ( policy --> max value )
    nextA = tf.argmax(currentQ, axis=1)
    # Fixed One State --> Next Four Action ==> Q Value
    nextQ = tf.placeholder(dtype=tf.float32, shape=[1, 4])
    # MSE
    # loss function = nextQ( reward + discount_factor * max q(s', a') ) - currentQ ( q(s, a) )
    cost = tf.reduce_sum(tf.square(nextQ - currentQ))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    g = 0.09
    e = 0.1
    num_episodes = 2000

    jList = []
    rList = []

    current_state = env.reset()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_episodes):
            total_reward_sum = 0
            done = False
            j = 0

            while j < 99:
                j += 1
                next_action, all_current_Q = sess.run([nextA, currentQ],
                                                      feed_dict={input1: np.identity(16)[current_state:current_state + 1]})
                if np.random.rand(1) < e:
                    next_action[0] = env.action_space.sample()

                next_state, reward, done, _ = env.step(next_action[0])
                all_next_Q = sess.run(currentQ,
                                      feed_dict={input1: np.identity(16)[next_state:next_state + 1]})

                # Q model update
                targetQ = all_current_Q
                targetQ[0, next_action[0]] = reward + g * np.max(all_next_Q)

                # train
                _ = sess.run(train,
                             feed_dict={input1: np.identity(16)[current_state:current_state + 1], nextQ: targetQ})
                total_reward_sum += reward
                current_state = next_state
                if done:
                    e = 1./((i/50) + 10)
                    break

            jList.append(j)
            rList.append(total_reward_sum)

    print('successful episodes:' + str(sum(rList) / num_episodes) + '%')
    plt.plot(rList)
    plt.figure()
    plt.plot(jList)
    plt.show()