import tensorflow as tf

if __name__ == "__main__":
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(a.get_shape())
    b = tf.constant([1, 0, 1])
    print(b.get_shape())
    b = tf.expand_dims(b, axis=1)
    print(b.get_shape())
    c = tf.matmul(a, b)

    sess = tf.Session()
    result = sess.run(c)
    print(result)
