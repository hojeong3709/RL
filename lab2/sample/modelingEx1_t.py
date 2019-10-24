import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
learning_rate = 0.1

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# hidden 1
w1 = tf.Variable(tf.random_normal([2,10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# hidden 2
w2 = tf.Variable(tf.random_normal([10,8]), name='weight2')
b2 = tf.Variable(tf.random_normal([8]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

# hidden 3
w3 = tf.Variable(tf.random_normal([8, 4]), name='weight3')
b3 = tf.Variable(tf.random_normal([4]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, w3) + b3)

# output
w4 = tf.Variable(tf.random_normal([4, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
logit = tf.matmul(layer3, w4) + b4
hypothesis = tf.sigmoid(logit)
cost = -tf.reduce_mean((y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis)))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        _cost, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        if step % 100 == 0:
            print('cost:{}'.format( _cost))

    _h, _p, _a = sess.run([hypothesis, prediction, accuracy],
                          feed_dict={x:x_data, y:y_data})
    print('hypothesis:\n{}\n\nprediction:\n{}\n\naccuracy:{}'.format(_h, _p, _a))


