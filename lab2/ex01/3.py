
import tensorflow as tf

if __name__ == "__main__":
    input_data = [1, 2, 3, 4, 5]
    x = tf.placeholder(dtype=tf.float32)
    y = x * 2

    sess = tf.Session()
    result = sess.run(y, feed_dict={x: input_data})
    print(result)

    input_data2 = [[1, 2], [3, 4]]
    a = tf.placeholder(dtype=tf.int32, shape=[2, 2], name="input_a")
    b = a * 2

    result = sess.run(b, feed_dict={a: input_data2})
    print(result)
