
import tensorflow as tf

if __name__ == "__main__":

    a = tf.constant([10], dtype=tf.float32, name="const_a")
    b = tf.constant([5], dtype=tf.float32, name="const_b")
    c = tf.constant([22], dtype=tf.float32, name="const_c")
    d = tf.constant([4], dtype=tf.float32, name="const_d")

    e = tf.multiply(a, b, name="multiple_e")
    f = tf.subtract(e, c, name="subtract_f")
    g = tf.div(f, d, name="division_g")

    sess = tf.Session()
    result = sess.run(g)

    print(result)

    tf.summary.FileWriter('./g_graph', sess.graph)

