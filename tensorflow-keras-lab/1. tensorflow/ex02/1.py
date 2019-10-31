
import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    tf.set_random_seed(777)
    learning_rate = 0.01

    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

    x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    w = tf.Variable(tf.random_normal([2, 1]), name="weight")
    b = tf.Variable(tf.random_normal([1]), name="bias")

    z = tf.matmul(x, w) + b
    p = tf.sigmoid(z)

    # logL(w) == l(w) --> binary cross entropy
    cost = - tf.reduce_mean((y * tf.log(p) + (1 - y) * tf.log(1 - p)))
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    prediction = tf.cast(p > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(1000):
            _w, _cost, _ = sess.run([w, cost, train], feed_dict={x: x_data, y: y_data})
            if step % 100 == 0:
                print("w: {}, cost: {}".format(_w, _cost))

        _h, _p, _a = sess.run([p, prediction, accuracy], feed_dict={x: x_data, y: y_data})
        print("h: {}\n, p: {}\n, a: {}".format(_h, _p, _a))