
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    learning_rate = 0.001
    training_epoches = 15
    batch_size = 10

    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    # 4 tensor ==> [ batch, width, height, channel ]
    x_img = tf.reshape(x, [-1, 28, 28, 1])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # kernel 크기를 알 수 없으므로 이것이 학습대상, 3 x 3 x 1 크기 32개 kernel 생성
    w1 = tf.Variable(tf.random.normal([3, 3, 1, 32], stddev=0.01))
    L1 = tf.nn.conv2d(x_img, w1, strides=[1, 1, 1, 1], padding='SAME')
    # L1 통과 후 나오는 결과값 모양 ==> (?, 28, 28, 32)
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # L1 max pool 통과 후 나오는 결과값 모양 ==> (?, 14, 14, 32)
    w2 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, w2, strides=[1, 1, 1, 1], padding='SAME')
    # (?, 14, 14, 64)
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # (?, 7, 7, 64)
    L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])

    w3 = tf.get_variable('w3', shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([10]))

    logit = tf.matmul(L2_flat, w3) + b
    hypothesis = tf.nn.softmax(logit)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    prediction = tf.argmax(hypothesis, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y, axis=1)), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epoches):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _c, _ = sess.run([cost, train], feed_dict={x: batch_x, y: batch_y})
                avg_cost = _c / total_batch

            print("epoch : {} cost : {}".format(epoch + 1, avg_cost))

        print("accuracy : {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))