
import tensorflow as tf

if __name__ == "__main__":
    input_data = [1, 2, 3, 4, 5]
    X = tf.placeholder(dtype=tf.float32)
    print(type(X))
    W = tf.Variable([2], dtype=tf.float32)
    print(type(W))
    y = W * X

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    C = tf.assign(W, [10])
    sess.run(C)

    result = sess.run(y, feed_dict={X: input_data})
    print(result)