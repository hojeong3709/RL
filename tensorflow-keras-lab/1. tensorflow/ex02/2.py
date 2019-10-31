
import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    tf.set_random_seed(777)
    learning_rate = 0.1

    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

    input_x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    target_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    # hidden_layer_1
    w1 = tf.Variable(tf.random_normal([2, 10]), name="w1")
    b1 = tf.Variable(tf.random_normal([10]), name="b1")

    layer1 = tf.sigmoid(tf.matmul(input_x, w1) + b1)

    # hidden_layer_2
    w2 = tf.Variable(tf.random_normal([10, 4]), name="w2")
    b2 = tf.Variable(tf.random_normal([4]), name="b2")

    layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

    # hidden_layer_3
    w3 = tf.Variable(tf.random_normal([4, 1]), name="w3")
    b3 = tf.Variable(tf.random_normal([1]), name="b3")

    # Fully_Connected_layer
    fc_layer = tf.sigmoid(tf.matmul(layer2, w3) + b3)

    # logL(w) == l(w) --> binary cross entropy
    cost = -tf.reduce_mean(target_y * tf.log(fc_layer) + (1 - target_y) * tf.log(1 - fc_layer))
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    prediction = tf.cast(fc_layer > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, target_y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(10000):
            _, _cost = sess.run([train, cost], feed_dict={input_x: x_data, target_y: y_data})
            if step % 100 == 0:
                print("cost: {}".format(_cost))

        _h, _p, _a = sess.run([fc_layer, prediction, accuracy], feed_dict={input_x: x_data, target_y: y_data})
        print("h: {}\n, p: {}\n, a: {}".format(_h, _p, _a))