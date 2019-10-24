
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    learning_rate = 0.001
    training_epoches = 15
    batch_size = 10

    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # 1. hidden layer 추가
    # 2. activation function 교체 ( sigmoid --> relu )
    # 3. xaiver initializer 적용

    # w1 = tf.Variable(tf.random_normal([784, 256]), name="weight1")
    w1 = tf.get_variable('w1', shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([256]), name="bias1")

    layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    # w2 = tf.Variable(tf.random_normal([256, 256]), name="weight2")
    w2 = tf.get_variable('w2', shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([256]), name="bias2")

    layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

    # w3 = tf.Variable(tf.random_normal([256, 10]), name="weight3")
    w3 = tf.get_variable('w3', shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([10]), name="bias3")

    logit = tf.matmul(layer2, w3) + b3
    hypothesis = tf.nn.softmax(logit)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    prediction = tf.argmax(hypothesis, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y, axis=1)), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epoches):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _c, _ = sess.run([cost, train], feed_dict={x: batch_x, y: batch_y})
                avg_cost = _c / total_batch

            print("epoch : {} cost : {}".format(epoch + 1, avg_cost))

        print("accuracy : {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))