import tensorflow as tf

g = tf.Graph()
with g.as_default():
    in1 = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='input1')
    in2 = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='input2')
    const = tf.constant(2, dtype=tf.float32, name='static_value')

    with tf.name_scope('main'):
        with tf.name_scope('Apart'):
            a_mul = tf.multiply(in1, const)
            a_out = tf.subtract(a_mul, in1)

        with tf.name_scope('Bpart'):
            b_mul = tf.multiply(in2, const)
            b_out = tf.subtract(b_mul, in2)

        with tf.name_scope('Cpart'):
            c_div = tf.div(a_out, b_out)
            c_out = tf.add(c_div, const)

        with tf.name_scope('Dpart'):
            d_div = tf.div(b_out, a_out)
            d_out = tf.add(d_div, const)
    out = tf.maximum(c_out, d_out)
    sess = tf.Session(graph=g)
    _result, _c_out, _d_out = sess.run([out, c_out, d_out],
                                       feed_dict={in1:[[7,3],[5,2]], in2:[[5,6],[8,1]]})
print('c_out:\n{}\n\nd_out:\n{}\n\nresult:\n{}'.format(_c_out, _d_out, _result))
tf.summary.FileWriter('./g_graph', graph=g)