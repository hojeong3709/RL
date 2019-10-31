
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

tf.set_random_seed(777)

if __name__ == "__main__":
    iris_data = pd.read_csv("data/Iris2.csv")
    data = iris_data.values
    x_data = data[:, 1:-1]
    y_data = data[:, [-1]]

    # y_data_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    # y_data = [y_data_dict[key] for key in y_data.flatten()]
    # y_one_hot = tf.reshape(tf.one_hot(y_data, 3), [-1, 3])
    #
    # print(type(y_data))
    # print(y_one_hot.get_shape())

    e = LabelEncoder()
    e.fit(y_data)
    y_data = e.transform(y_data)
    print(y_data.shape)
    y_data = np.expand_dims(y_data, axis=1)
    print(y_data.shape)

    learning_rate = 0.05
    training_epoches = 15
    batch_size = 10

    x = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    y = tf.placeholder(dtype=tf.int32, shape=[None, 1])

    y_one_hot = tf.one_hot(y, 3)
    y_one_hot = tf.reshape(y_one_hot, [-1, 3])

    w = tf.Variable(tf.random_normal([4, 3]), name="w")
    b = tf.Variable(tf.random_normal([3]), name="b")

    logit = tf.matmul(x, w) + b
    hypothesis = tf.nn.softmax(logit)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y_one_hot))

    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    prediction = tf.equal(tf.argmax(hypothesis, axis=1), tf.argmax(y_one_hot, axis=1))
    accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(2000):
            sess.run(train, feed_dict={x: x_data, y: y_data})
            if step % 100 == 0:
                _p, _a, _c, _ = sess.run([prediction, accuracy, cost, train], feed_dict={x: x_data, y: y_data})
                print("a : {} c : {}".format(_a, _c))
