import tensorflow as tf

if __name__ == "__main__":
    print(tf.get_default_graph())
    g = tf.Graph()
    print(g)

    a = tf.constant(5)
    print(a.graph is g)
    print(a.graph is tf.get_default_graph())

    with g.as_default():
        b = tf.constant(10)
        c = tf.multiply(b, 5)
        d = tf.subtract(c, 7)

        print(b.graph is g)

    # session default
    sess = tf.Session(graph=tf.get_default_graph())

    # session others
    sess2 = tf.Session(graph=g)

    print(sess2.run(d))

    tf.summary.FileWriter("./g_graph", sess2.graph)
