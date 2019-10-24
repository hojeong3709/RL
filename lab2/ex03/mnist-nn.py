
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.set_random_seed(777)

if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    learning_rate = 0.001
    training_epoches = 15
    batch_size = 10

    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    w = tf.Variable(tf.random_normal([784, 10]), name="weight")
    b = tf.Variable(tf.random_normal([10]), name="bias")

    logit = tf.matmul(x, w) + b
    hypothesis = tf.nn.softmax(logit)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y))
    train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

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

        print("accuracy : {}".format(sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})))