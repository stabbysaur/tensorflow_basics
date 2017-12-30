import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

import utils

PATH = 'C:\\Users\\Stabby\\Documents\\Python\\Tensorflow\\'
LOG_DIR = PATH + 'logs\\simple_nn\\'
DATA_DIR = PATH + 'data\\'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

learning_rate = 0.1
decay_rate = 0.97
epochs = 500
batch_size = 128

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 28*28
n_classes = 10

"""set up inputs"""
X = tf.placeholder(tf.float32, shape=[None, n_input])  # 'None' allows variable  batch size!
y = tf.placeholder(tf.float32, shape=[None, n_classes])

"""set up weights"""
weights = {'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
           'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
           'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))}

biases = {'b1': tf.Variable(tf.random_normal([n_hidden_1])),
          'b2': tf.Variable(tf.random_normal([n_hidden_2])),
          'out': tf.Variable(tf.random_normal([n_classes]))}

lr = tf.Variable(initial_value=learning_rate,
                 trainable=False)

def mini_NN(X):

    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out

with tf.name_scope("Model"):
    logits = mini_NN(X)
with tf.name_scope("Loss"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                   logits=logits))
    correct_preds = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    opt_step = optimizer.minimize(cost)

init = tf.global_variables_initializer()

"""summaries"""
tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("learning_rate", lr)
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:

    sess.run(init)
    summary_writer = tf.summary.FileWriter(logdir=LOG_DIR, graph=tf.get_default_graph())

    for epoch in range(epochs):
        print("Epoch: {0}".format(epoch))

        batch_X, batch_y = mnist.train.next_batch(batch_size)
        _, c, acc, summary = sess.run([opt_step, cost, accuracy, merged_summary_op],
                                 feed_dict={X: batch_X, y: batch_y, lr: learning_rate * (decay_rate ** epoch)})

        summary_writer.add_summary(summary, global_step=epoch)
        print("Cost: {0}, Accuracy: {1}".format(c, acc))

    print("Testing accuracy: {0}".format(sess.run(accuracy, feed_dict={X: mnist.test.images,
                                                                       y: mnist.test.labels})))

utils.tensorboard_output(LOG_DIR)