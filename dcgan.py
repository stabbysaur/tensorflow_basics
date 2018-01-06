import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import utils

PATH = 'C:\\Users\\Stabby\\Documents\\Python\\Tensorflow\\'
LOG_DIR = PATH + 'logs\\simple_lstm\\'
CHECK_DIR = PATH + 'checkpoints\\dcgan\\'
DATA_DIR = PATH + 'data\\'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

"""wipe out log_path, create checkpoint path"""
utils.wipe_dir(LOG_DIR)
utils.make_dir(CHECK_DIR)

"""
Notes on training DCGAN (+GANs in general)! 

-- Use batchnorm and dropout 
-- Leaky ReLU instead of ReLU
-- Normalize inputs between +/-1
-- Construct different batches for real and fake (train all real or all fake) 
-- Downsampling: replace pooling with stride 
-- Upsampling: pixelshuffle, convtranspose2d + stride 
-- Use soft labels: 0.7-1.0 instead of 1.0, 0.0-0.3 instead of 0.0 
-- Noisy labels for the discriminator (occasionally flip) 
-- SGD for discriminator, ADAM for generator 
-- Don't try to find a clever schedule for D/G updates. 
---- Try something like: if lossD > A, update D. If lossG > B, update G. 
-- Auxiliary GANs --> allow discriminator to also learn labels (like the digits of MNIST) 
-- Add some noise to inputs of D, decay over time 
-- Also add some Gaussian noise to every layer of G 

Discriminator: Convolutions --> Dense + sigmoid 
Generator: Dense --> Transpose (inverse) convolutions --> tanh
-- If dense layer is too large, may cause mode collapse. If left out, won't learn. 
-- Final tanh helps with stability 
"""

"""parameters"""
learning_rate = 0.0001
decay_rate = 0.97
epochs = 50
batch_size = 125
gen_threshold = 3.0
disc_threshold = 0.25
use_saved = True

image_dim = 28*28  # horizonal pixels are features!
noise_dim = 100

"""placeholders"""
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

lr = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)  # batchnorm flag

def generator(X, reuse=False):

    """conv2d_transpose takes in [input, filters (output), kernel_size...]"""

    with tf.variable_scope('Generator', reuse=reuse):

        """dense --> deconvolutions"""
        X = tf.layers.dense(X, units=7*7*128)
        X = tf.layers.batch_normalization(X, training=is_training)
        X = tf.nn.relu(X)

        """reshape to include an axis for channels"""
        X = tf.reshape(X, shape=[-1, 7, 7, 128])

        """deconvolution: (batch_size, 14, 14, 64)"""
        X = tf.layers.conv2d_transpose(X, 64, 5, strides=2, padding='same')
        X = tf.layers.batch_normalization(X, training=is_training)
        X = tf.nn.relu(X)

        """deconvolution: (batch_size, 28, 28, 1) = image shape"""
        X = tf.layers.conv2d_transpose(X, 1, 5, strides=2, padding='same')
        X = tf.nn.tanh(X)
        return X

def discriminator(X, reuse=False):

    with tf.variable_scope('Discriminator', reuse=reuse):

        X = tf.layers.conv2d(X, 64, 5, strides=2, padding='same')
        X = tf.layers.batch_normalization(X, training=is_training)
        X = tf.nn.leaky_relu(X)

        X = tf.layers.conv2d(X, 128, 5, strides=2, padding='same')
        X = tf.layers.batch_normalization(X, training=is_training)
        X = tf.nn.leaky_relu(X)

        X = tf.reshape(X, shape=[-1, 7*7*128])
        X = tf.layers.dense(X, 1024)
        X = tf.layers.batch_normalization(X, training=is_training)
        X = tf.nn.leaky_relu(X)

        X = tf.layers.dense(X, 2)  # real / false
        return X

"""model!"""
gen_sample = generator(noise_input)

disc_real = discriminator(image_input)
disc_fake = discriminator(gen_sample, reuse=True)

stacked_gan = discriminator(gen_sample, reuse=True)

# """loss (using soft labels)!"""
# disc_loss_real = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_real,
#                                                                 labels=tf.random_uniform(shape=[batch_size], minval=0.7, maxval=1.0))
# disc_loss_fake = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake,
#                                                                 labels=tf.random_uniform(shape=[batch_size], minval=0.0, maxval=0.3))

"""loss"""
disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_real,
                                                                labels=tf.ones(shape=[batch_size], dtype=tf.int32)))
disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake,
                                                                labels=tf.zeros(shape=[batch_size], dtype=tf.int32)))


disc_loss = disc_loss_real + disc_loss_fake
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=stacked_gan,
                                                          labels=tf.ones([batch_size], dtype=tf.int32)))

"""optimizers"""
gen_opt = tf.train.AdamOptimizer(beta1=0.05, beta2=0.999, learning_rate=lr)
# disc_opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
disc_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999)


"""specify the variables that should be modified by each optimizer"""
gen_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                             scope='Generator')
disc_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                              scope='Discriminator')

"""
Separate the UPDATE_OPS collection. 
This contains all batchnorm operations, which must be run before backprop (minimize). 
To order operations, use the control_dependencies wrapper.
"""
gen_update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS,
                                   scope='Generator')
with tf.control_dependencies(gen_update_ops):
    train_gen = gen_opt.minimize(gen_loss, var_list=gen_vars)

disc_update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS,
                                    scope='Discriminator')
with tf.control_dependencies(disc_update_ops):
    train_disc = disc_opt.minimize(disc_loss, var_list=disc_vars)

"""DON'T FEEL LIKE TENSORBOARDING >:["""
init = tf.global_variables_initializer()

restored = False
if use_saved:
    try:
        ckpt = tf.train.get_checkpoint_state(CHECK_DIR)
        if ckpt is not None:
            restored = True
    except Exception as err:
        print("Training from scratch!")
        print(err)

with tf.Session() as sess:

    sess.run([init])
    saver = tf.train.Saver(tf.global_variables())

    if restored:
        saver.restore(sess, ckpt.model_checkpoint_path)

    samples = len(mnist.train.images)
    batches = samples // batch_size

    """initialize losses to check against the thresholds"""
    last_disc_loss = 1000.
    last_gen_loss = 1000.

    for epoch in range(epochs):

        current_rate = learning_rate * (decay_rate ** epoch)

        print("Epoch {0}".format(epoch))

        for batch in range(batches):
            batch_x, _ = mnist.train.next_batch(batch_size)
            batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])

            """rescale to [-1, 1]"""
            batch_x = (batch_x * 2.) - 1.

            """discriminator step"""
            noise = np.random.uniform(-1., 1., size=(batch_size, noise_dim))

            trained_something = False
            if last_disc_loss > disc_threshold:
                _, dl = sess.run([train_disc, disc_loss],
                                 feed_dict={image_input: batch_x, noise_input: noise, is_training: True, lr: current_rate})
                last_disc_loss = dl
                trained_something = True
            else:
                dl = sess.run(disc_loss,
                               feed_dict={image_input: batch_x, noise_input: noise, is_training: True, lr: current_rate})
                last_disc_loss = dl

            """generator step"""
            noise = np.random.uniform(-1., 1., size=(batch_size, noise_dim))

            if (last_gen_loss > gen_threshold) or not trained_something:
                _, gl = sess.run([train_gen, gen_loss],
                                 feed_dict={noise_input: noise, is_training: True, lr: current_rate})
                last_gen_loss = gl
            else:
                gl = sess.run(gen_loss,
                              feed_dict={noise_input: noise, is_training: True, lr: current_rate})
                last_gen_loss = gl

            if batch % 20 == 0:
                print("Epoch {0}, batch {1}, generator cost {2}, discriminator cost {3}".format(epoch, batch, gl, dl))

        saver.save(sess, CHECK_DIR + 'dcgan.ckpt',
                   global_step=epoch * batches)

    n = 6
    canvas = np.empty((28 * n, 28 * n))
    for i in range(n):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[n, noise_dim])
        # Generate image from noise.
        g = sess.run(gen_sample, feed_dict={noise_input: z, is_training: False})
        # Rescale values to the original [0, 1] (from tanh -> [-1, 1])
        g = (g + 1.) / 2.
        # Reverse colours for better display
        g = -1 * (g - 1)
        for j in range(n):
            # Draw the generated digits
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

    plt.figure(figsize=(n, n))
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.show()