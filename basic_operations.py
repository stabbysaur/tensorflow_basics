import tensorflow as tf

"""operations with constants"""
a = tf.constant(5)
b = tf.constant(10)

with tf.Session() as sess:
    print("Sum: {0}".format(sess.run(a + b)))
    print("Product: {0}".format(sess.run(a * b)))

"""operations with variables"""
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

with tf.Session() as sess:
    sum_op = tf.add(a, b)
    prod_op = tf.multiply(a, b)

    print("Sum: {0}".format(sess.run(sum_op, feed_dict={a: 5, b:10})))
    print("Prodcut: {0}".format(sess.run(prod_op, feed_dict={a: 5, b:10})))