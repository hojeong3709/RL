import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print(tf.get_default_graph())
g = tf.Graph()
print(g)

a = tf.constant(5)
print(a.graph is g)
print(a.graph is tf.get_default_graph())

with g.as_default():
    b = tf.multiply(2,4)
    c = tf.add(b,5)
    d = tf.subtract(c, 7)
    print(b.graph is g)

sess = tf.Session(graph=tf.get_default_graph())
sess2 = tf.Session(graph=g)
print('result:',sess2.run(d))
tf.summary.FileWriter('./g_graph', graph=g)

