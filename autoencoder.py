import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import utils

PATH = 'C:\\Users\\Stabby\\Documents\\Python\\Tensorflow\\'
LOG_DIR = PATH + 'logs\\simple_lstm\\'
DATA_DIR = PATH + 'data\\'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

"""wipe out log_path"""
utils.wipe_dir(LOG_DIR)

"""
Intution: 
-- dimensionality reduction! 
-- encoder maps inputs into a lower dimension 
-- decoder reverses the mapping back to the original input size 
-- the cost is the difference between the original and the decoded input!
-- minimizing cost therefore minimizes compression loss 
"""

learning_rate = 0.001
decay_rate = 1.00  # really for epochs, not training steps
training_steps = 10000
batch_size = 128

num_inputs = 28*28  # horizonal pixels are features!
num_hidden_1 = 256
num_hidden_2 = 128

"""inputs"""
X = tf.placeholder(tf.float32, [None, num_inputs])
lr = tf.placeholder(tf.float32)

"""weights"""
weights = {'encoder_1': tf.Variable(initial_value=tf.random_normal([num_inputs, num_hidden_1])),
           'encoder_2': tf.Variable(initial_value=tf.random_normal([num_hidden_1, num_hidden_2])),
           'decoder_1': tf.Variable(initial_value=tf.random_normal([num_hidden_2, num_hidden_1])),
           'decoder_2': tf.Variable(initial_value=tf.random_normal([num_hidden_1, num_inputs]))}

biases = {'encoder_1': tf.Variable(initial_value=tf.random_normal([num_hidden_1])),
          'encoder_2': tf.Variable(initial_value=tf.random_normal([num_hidden_2])),
          'decoder_1': tf.Variable(initial_value=tf.random_normal([num_hidden_1])),
          'decoder_2': tf.Variable(initial_value=tf.random_normal([num_inputs]))}

def encoder(X, weights, biases):

    layer1 = tf.add(tf.matmul(X, weights['encoder_1']), biases['encoder_1'])
    layer1 = tf.nn.sigmoid(layer1)

    layer2 = tf.add(tf.matmul(layer1, weights['encoder_2']), biases['encoder_2'])
    layer2 = tf.nn.sigmoid(layer2)

    return layer2

def decoder(X, weights, biases):

    layer1 = tf.add(tf.matmul(X, weights['decoder_1']), biases['decoder_1'])
    layer1 = tf.nn.sigmoid(layer1)

    layer2 = tf.add(tf.matmul(layer1, weights['decoder_2']), biases['decoder_2'])
    layer2 = tf.nn.sigmoid(layer2)

    return layer2

"""model!"""
with tf.name_scope("Model"):
    preds = decoder(encoder(X, weights, biases), weights, biases)
    targets = X

"""cost!"""
with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.pow(targets - preds, 2))

"""optimizer!"""
with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)

"""summaries!"""
tf.summary.scalar("loss", loss)
tf.summary.scalar("learning_rate", lr)
merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

"""training loop!"""
with tf.Session() as sess:

    sess.run(init)
    summary_writer = tf.summary.FileWriter(logdir=LOG_DIR, graph=tf.get_default_graph())

    for step in range(training_steps):
        batch_x, _ = mnist.train.next_batch(batch_size)

        _, c, summary = sess.run([train_op, loss, merged_summary_op],
                                 feed_dict={X:batch_x, lr:learning_rate})

        summary_writer.add_summary(summary)
        if step % 100 == 0:
            print("Step: {0}, Cost: {1}".format(step, c))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(preds, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the generated digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the generated digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()