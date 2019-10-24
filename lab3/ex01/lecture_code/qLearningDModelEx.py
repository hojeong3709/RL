import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
tf.set_random_seed(777)

input1 = tf.placeholder(dtype=tf.float32, shape=[1, 16])
w = tf.Variable(tf.random_normal([16,4], stddev=0.01))
Qout = tf.matmul(input1, w) #q value
predict = tf.argmax(Qout, axis=1) #action

nextQ = tf.placeholder(dtype=tf.float32, shape=[1,4])
cost = tf.reduce_sum(tf.square(nextQ - Qout))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

g = 0.99
e = 0.1
num_episodes = 2000

jList = []
rList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        current_state = env.reset()
        rewardAll = 0
        done = False
        j = 0

        while j < 99:
            j += 1
            next_action, allQ = sess.run([predict, Qout],
                                feed_dict={input1:np.identity(16)[current_state:current_state + 1]})

            if np.random.rand(1) < e:
                next_action[0] = env.action_space.sample()

            next_state, reward, done, _ = env.step(next_action[0])
            Q1 = sess.run(Qout, feed_dict={input1:np.identity(16)[next_state: next_state + 1]})
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, next_action[0]] = reward + g * maxQ1
            _ = sess.run(train, feed_dict={input1:np.identity(16)[current_state:current_state + 1],
                                           nextQ:targetQ})
            rewardAll += reward
            current_state = next_state
            if done == True:
                e = 1./((i/50)+10)
                break
        jList.append(j)
        rList.append(rewardAll)
print('succesful episodes:'+str(sum(rList)/ num_episodes)+'%')
plt.plot(rList)
plt.figure()
plt.plot(jList)
plt.show()
