import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
tf.set_random_seed(777)

iris_data = pd.read_csv('Iris2.csv')
data_set = iris_data.values
x_data = data_set[:, 1:5]
y_data = data_set[:, 5]

e = LabelEncoder()
e.fit(y_data)
y_data = e.transform(y_data)
print(y_data.shape)
y_data = np.expand_dims(y_data, axis=1)
print(y_data.shape)

x = tf.placeholder(dtype=tf.float32, shape=[None, 4])
y = tf.placeholder(dtype=tf.int32, shape=[None,1])

y_one_hot = tf.one_hot(y, 3)
y_one_hot = tf.reshape(y_one_hot, [-1, 3])

w = tf.Variable(tf.random_normal([4,3]),name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')

logit = tf.matmul(x, w) + b
hypothesis = tf.nn.softmax(logit)

cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y_one_hot)
cost = tf.reduce_mean(cost_i)
train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

correct_prediction = tf.equal(tf.argmax(hypothesis, axis=1),
                              tf.argmax(y_one_hot, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(3001):
        sess.run(train, feed_dict={x:x_data, y:y_data})
        if step %100 == 0:
            _c, _a = sess.run([cost, accuracy], feed_dict={x:x_data, y:y_data})
            print('step:{}  cost:{} accuracy:{}'.format(step, _c, _a))

    print('accuracy:', sess.run(accuracy, feed_dict={x:x_data, y:y_data}))
