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
Intuition notes! 

Generator: Takes in noise ([noise_dim]), outputs MNIST-like thing ([num_inputs]) 
Discriminator: Takes in MNIST-like thing ([num_inputs]), outputs real/fake ([1])

Keep optimizers and variables separate for each! 
Train the discriminator a few steps first, allow it to stay ahead of the generator. 

Mode collapse: The generator isn't incentivized to capture all modes of the distribution, 
just one part of it that can be used to trick the discriminator. Therefore a multimodal 
underlying distribution might get squashed into a single mode.
-- AdaGAN: Stack multiple GANs for the different modes. (Slow!) 
-- Experience replay: Show the discriminator old samples to prevent the generator from hopping
between modes

...and more!
"""

learning_rate = 0.0002
decay_rate = 0.97
epochs = 100
batch_size = 125
gen_disc_ratio = 2  # how many times should the discriminator be trained compared to the generator?

num_inputs = 28*28  # horizonal pixels are features!
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100

def glorot_init(shape):

    """
    Enabled the move away from generative pre-training!

    Assuming a linear neuron, this keeps the variance of the input and output gradients the same.
    Linear assumption works even with nonlinear activations, because upon initialization the
    areas explored are close to zero (where tanh, sigm etc all look locally linear).
    """

    return tf.random_normal(shape=shape, stddev=tf.sqrt(2. / (shape[0] + shape[1])))

weights = {'gen_hidden_1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
           'gen_out': tf.Variable(glorot_init([gen_hidden_dim, num_inputs])),
           'disc_hidden_1': tf.Variable(glorot_init([num_inputs, disc_hidden_dim])),
           'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1]))}

biases = {'gen_bias_1': tf.Variable(tf.zeros([gen_hidden_dim])),
          'gen_bias_out': tf.Variable(tf.zeros([num_inputs])),
          'disc_bias_1': tf.Variable(tf.zeros([disc_hidden_dim])),
          'disc_bias_out': tf.Variable(tf.zeros([1]))}

"""placeholders"""
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, num_inputs], name='disc_input')
lr = tf.placeholder(tf.float32, name='learning_rate')

def generator(X):

    hidden_layer = tf.add(tf.matmul(X, weights['gen_hidden_1']), biases['gen_bias_1'])
    hidden_layer = tf.nn.relu(hidden_layer)

    out_layer = tf.add(tf.matmul(hidden_layer, weights['gen_out']), biases['gen_bias_out'])
    return tf.nn.sigmoid(out_layer)

def discriminator(X):

    hidden_layer = tf.add(tf.matmul(X, weights['disc_hidden_1']), biases['disc_bias_1'])
    hidden_layer = tf.nn.relu(hidden_layer)

    out_layer = tf.add(tf.matmul(hidden_layer, weights['disc_out']), biases['disc_bias_out'])
    return tf.nn.sigmoid(out_layer)

"""networks!"""
with tf.name_scope("Model"):
    gen_output = generator(gen_input)

    """parameters are shared"""
    disc_real = discriminator(disc_input)
    disc_fake = discriminator(gen_output)

with tf.name_scope("Cost"):

    """
    We want to maximize --> minimize the negation. 
    
    Generator: Maximize the discriminator's probability of a real sample when the input is fake (disc_fake).
    Discriminator: Maximize how well it classifies real and fake samples. 
    -- Real: disc_real 
    -- Fake: (1 - disc_fake) 
    (Remember, discriminator output is probability that a sample is real.) 
    
    Use log-likelihoods.
    """

    gen_cost = -tf.reduce_mean(tf.log(disc_fake))
    disc_cost = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

"""split out variables -- these should be optimized based on EITHER the disc or gan cost, not both"""
gen_weights = [weights['gen_hidden_1'], weights['gen_out'], biases['gen_bias_1'], biases['gen_bias_out']]
disc_weights = [weights['disc_hidden_1'], weights['disc_out'], biases['disc_bias_1'], biases['disc_bias_out']]

with tf.name_scope("Optimizer"):

    gen_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(gen_cost, var_list=gen_weights)
    disc_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(disc_cost, var_list=disc_weights)

with tf.name_scope("Summary"):

    tf.summary.scalar("gen_cost", gen_cost)
    tf.summary.scalar("disc_cost", disc_cost)
    tf.summary.scalar("learning_rate", lr)
    merged_summary_opt = tf.summary.merge_all()

init = tf.global_variables_initializer()

"""train!"""
with tf.Session() as sess:

    sess.run(init)
    summary_writer = tf.summary.FileWriter(logdir=LOG_DIR, graph=tf.get_default_graph())

    for epoch in range(epochs):

        samples = len(mnist.train.images)
        batches = samples // batch_size
        current_rate = learning_rate * (decay_rate ** epoch)

        print("Epoch {0}".format(epoch))

        for batch in range(batches):
            batch_x, _ = mnist.train.next_batch(batch_size)
            noise = np.random.uniform(-1., 1., size=(batch_size, noise_dim))

            feed_dict = {disc_input: batch_x, gen_input: noise, lr: current_rate}

            """train discriminator. every gen_disc_ratio steps, train generator as well."""
            if batch % gen_disc_ratio == 0:
                _, _, gc, dc, summary = sess.run([gen_opt, disc_opt, gen_cost, disc_cost, merged_summary_opt],
                                                 feed_dict=feed_dict)

            else:
                _, gc, dc, summary = sess.run([disc_opt, gen_cost, disc_cost, merged_summary_opt],
                                              feed_dict=feed_dict)

            summary_writer.add_summary(summary, global_step=(epoch * batches) + batch)
            if batch % 20 == 0:
                print("Epoch {0}, batch {1}, generator cost {2}, discriminator cost {3}".format(epoch, batch, gc, dc))

    # Testing
    # Generate images from noise, using the generator network.
    n = 6
    canvas = np.empty((28 * n, 28 * n))
    for i in range(n):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[n, noise_dim])
        # Generate image from noise.
        g = sess.run(gen_output, feed_dict={gen_input: z})
        # Reverse colours for better display
        g = -1 * (g - 1)
        for j in range(n):
            # Draw the generated digits
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

    plt.figure(figsize=(n, n))
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.show()

    utils.tensorboard_output(LOG_DIR)