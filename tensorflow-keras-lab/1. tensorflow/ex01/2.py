
import tensorflow as tf

if __name__ == "__main__":
    # vector
    # a = tf.constant([[5, 3], [2, 4]], dtype=tf.float32, name="input_a")
    a = tf.placeholder(dtype=tf.float32, shape=[2, 2], name="input_a")
    # scalar
    b = tf.reduce_sum(a, name="sum_b")
    # scalar
    c = tf.reduce_prod(a, name="prod_c")
    d = tf.add(b, c, name="add_d")
    sess = tf.Session()
    print(sess.run(d, feed_dict={a: [[4, 5], [2, 3]]}))

    print("a shape:", a.get_shape())

    e = tf.constant(10)
    print("e shape:", e.get_shape())
    print("e dtype:", e.dtype)

    f = tf.constant([10, 10])
    print("f shape:", f.get_shape())
    print("f dtype:", f.dtype)

    f = tf.cast(f, tf.int64)
    print("f dtype:", f.dtype)

    tf.summary.FileWriter("./g_graph", sess.graph)


