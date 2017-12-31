import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import utils

PATH = 'C:\\Users\\Stabby\\Documents\\Python\\Tensorflow\\'
LOG_DIR = PATH + 'logs\\simple_lstm\\'
DATA_DIR = PATH + 'data\\'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

"""wipe out log_path"""
utils.wipe_dir(LOG_DIR)

learning_rate = 0.001
decay_rate = 1.00  # CAREFUL WITH DECAY RATE -- cross validate
training_steps = 5000
batch_size = 128

num_inputs = 28  # horizonal pixels are features!
timesteps = 28  # timesteps -- each row of the MNIST image is a timestep
num_hidden = 128  # hidden units H
num_classes = 10

"""inputs"""
X = tf.placeholder(tf.float32, shape=[None, timesteps, num_inputs])
y = tf.placeholder(tf.float32, shape=[None, num_classes])
lr = tf.placeholder(tf.float32)

"""weights -- 2x hidden since forward and backward outputs are concatenated"""
weights = {'out': tf.Variable(initial_value=tf.random_normal([2*num_hidden, num_classes]))}
biases = {'out': tf.Variable(initial_value=tf.random_normal([num_classes]))}

def BiLSTM(X, weights, biases):

    """unstack to match tf's RNN format: (timesteps, batch_size, num_inputs)"""
    X = tf.unstack(X, timesteps, 1)  # input, length of axis, axis

    forward_lstm = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    backward_lstm = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    """_ is for the outputs of the forward and backward lstms"""
    outputs, _, _ = rnn.static_bidirectional_rnn(forward_lstm, backward_lstm, X, dtype=tf.float32)
    return tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])

"""graph!"""
with tf.name_scope("Model"):
    logits = BiLSTM(X, weights, biases)
    predictions = tf.nn.softmax(logits)

"""loss!"""
with tf.name_scope("Loss"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, axis=1), tf.argmax(y, axis=1)), tf.float32))

"""optimizer!"""
with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(cost)

"""summaries!"""
tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("learning_rate", lr)
merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

"""training loop"""
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logdir=LOG_DIR, graph=tf.get_default_graph())

    for step in range(training_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_inputs))

        feed_dict = {X: batch_x,
                     y: batch_y,
                     lr: learning_rate}

        _, c, acc, summary = sess.run([train_op, cost, accuracy, merged_summary_op],
                                      feed_dict=feed_dict)
        summary_writer.add_summary(summary, global_step=step)
        if step % 100 == 0:
            print("Step: {0}, cost: {1}, accuracy: {2}".format(step, c, acc))

    """testing accuracy"""
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_inputs))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, y: test_label}))

utils.tensorboard_output(LOG_DIR)