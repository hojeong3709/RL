import tensorflow as tf

with tf.name_scope('Scope_first'):
    a = tf.add(1,3, name="first_a")
    b = tf.multiply(a,3, name='first_b')

with tf.name_scope('Scope_second'):
    c = tf.add(4,5, name='second_c')
    d = tf.multiply(c,7, name='second_d')

e = tf.add(b, d)
sess = tf.Session()
print(sess.run(e))
tf.summary.FileWriter('./g_graph', graph=tf.get_default_graph())