import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

data = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = data[:, 0:-1]
y_data = data[:, [-1]]

x = tf.placeholder(dtype=tf.float32, shape=[None, 16])
y = tf.placeholder(dtype=tf.int32, shape=[None, 1])
y_ont_hot = tf.one_hot(y, 7)
print(y_ont_hot.get_shape())
y_ont_hot = tf.reshape(y_ont_hot, [-1, 7])
print(y_ont_hot.get_shape())

w = tf.Variable(tf.random_normal([16, 7]), name='weight')
b = tf.Variable(tf.random_normal([7]), name='bias')

logit = tf.matmul(x, w) + b
hypothesis = tf.nn.softmax(logit)
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y_ont_hot)
cost = tf.reduce_mean(cost_i)

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_ont_hot, axis=1)), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        _, _cost, _a, _p = sess.run([train, cost, accuracy, prediction], feed_dict={x:x_data, y:y_data})
        if step % 100 == 0:
            print("step:{}\nprediction:\n{}\n\ncost:{}\naccuracy:{}".format(step, _p, _cost, _a))

    _p = sess.run(prediction, feed_dict={x:x_data})
    for p, y in zip(_p, y_data):
        print('prediction:{}   target:{}'.format(p, y))