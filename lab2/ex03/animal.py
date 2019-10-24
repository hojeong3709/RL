
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

if __name__ == "__main__":
    data = np.loadtxt("data/data-04-zoo.csv", delimiter=",", dtype=np.float32)
    # matrix
    x_data = data[:, 0:-1]

    # matrix
    # y_data = data[:, -1]
    # vector
    y_data = data[:, [-1]]

    print(x_data.shape)
    print(y_data.shape)

    x = tf.placeholder(dtype=tf.float32, shape=[None, 16])
    y = tf.placeholder(dtype=tf.int32, shape=[None, 1])

    # one hot encoding을 수행하면 Shape가 [?, 1] --> [?, 7] 로 변환되길 기대하지만 다르게 변형됨.
    # [?, 1] --> [?, 1, 7]로 변화므로 reshape를 통해서 [?, 7]로 변경해줘야 한다.
    # one hot encoding은 3차원 tensor가 default 이다. why? RNN
    # y_one_hot = tf.one_hot(y, 7)
    # print(y_one_hot.get_shape())

    y_one_hot = tf.reshape(tf.one_hot(y, 7), [-1, 7])
    print(type(y_one_hot))
    print(y_one_hot.get_shape())

    w = tf.Variable(tf.random_normal([16, 7]), name="weight")
    b = tf.Variable(tf.random_normal([7]), name="bias")

    logit = tf.matmul(x, w) + b
    hypothesis = tf.nn.softmax(logit)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y_one_hot))

    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
    prediction = tf.argmax(hypothesis, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_one_hot, axis=1)), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(2000):
            _p, _a, _c, _ = sess.run([prediction, accuracy, cost, train], feed_dict={x: x_data, y: y_data})
            print("p: {} a : {} c : {}".format(_p, _a, _c))



