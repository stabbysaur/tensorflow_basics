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
epochs = 10000
batch_size = 128

num_inputs = 28  # horizonal pixels are features!
timesteps = 28  # timesteps -- each row of the MNIST image is a timestep
num_hidden = 128  # hidden units H
num_classes = 10

def sigmoid(X):

    """placeholder for sigmoid activation"""

    pass

def lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b):

    """
    Input data has dimension D.
    Hidden state has dimension H.
    Batch size is N.

    :param X: Input data NxD (batch x dim)
    :param prev_h: Previous hidden state NxH (batch x hidden dim)
    :param prev_c: Previous cell state NxH (batch x hidden dim).
    :param Wx: Input-to-hidden weights Dx4H (input dim x hidden (4 gates))
    :param Wh: Hidden-to-hidden weights Hx4H (hidden x hidden (4 gates))
    :param b: Biases 4H (hidden (4 gates))
    :return: next_h, next_c, cache
    """

    H = Wh.shape[0]  # dimension of hidden state
    a = X.dot(Wx) + prev_h.dot(Wh) + b  # (input*weights + hidden*hidden weights + bias)

    """
    calculate gates and activations.
    
    all of the gates depend on a weighted sum of: 
    -- input X 
    -- previous hidden state prev_h 
    -- bias b
    
    z_i: input gate -- how should we scale the input candidates?
    z_f: forget gate -- what part of the last cell state should we forget? 
    z_o: output gate -- what part of the updated cell state should we output? 
    
    The gates use sigmoids [0, 1]. 
    They can choose to let anything in the range of [nothing (0) to everything (1)] through.
    These aren't "weights" -- they are a measure of how much information to keep / forget.
    
    z_g: This uses tanh [-1, 1]. 
    Allowing the hidden state to shift in both directions helps with exploding/vanishing gradients.
    This creates new "candidates" (weighted activations of input/hidden) to add to the cell state.
    
    The next cell state is a combination of the old one and the input. Cell memory is recurrent!
    First it forgets part of the previous one, then adds the input gate --> input candidates. 
    c[t] = z_f*c[t-1] + z_i*z_g
    
    NOTE: The derivative of the next cell state with respect to the previous one is just z_f. 
    
    In the case of regular RNNs, backprop relies on the derivation of an activation function. 
    Over time, repeat multiplication causes gradients to vanish or explode. 
    
    For the LSTM, the derivative is the forget gate. If the forget gate = 1., then the deriv
    is the identity and won't have bad behavior in either direction.
    
    This is called the constant/linear error carousel!! Wheeeee. 
    
    The hidden state is equivalent to the output and depends on the updated cell memory. 
    
    Variants include peephole connections, where every gate has access to the previous cell state,
    and GRUs, where forget + input = update gate, and cell state = hidden state.
    """

    z_i = sigmoid(a[:,:H])  # input gate
    z_f = sigmoid(a[:,H:2*H])  # forget gate
    z_o = sigmoid(a[:,2*H:3*H])  # output gate
    z_g = np.tanh(a[:,3*H:])  # candidates

    next_c = z_f*prev_c + z_i*z_g
    z_t = np.tanh(next_c)
    next_h = z_o*z_t
    cache = (z_i, z_f, z_o, z_g, z_t, prev_c, prev_h, Wx, Wh, X)

    return next_h, next_c, cache

"""set up input placeholders"""
X = tf.placeholder(tf.float32, shape=[None, timesteps, num_inputs])
y = tf.placeholder(tf.float32, shape=[None, num_classes])
lr = tf.placeholder(tf.float32)

"""weights -- define for the output layer only"""
weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

def RNN(X, weights, biases):

    """
    input data has shape (batch_size, timesteps, num_inputs).
    tensorflow RNN requires (timesteps, batch_size, num_inputs). REQUIRES!
    unstack before running!

    NOTES: this sort of process is fine for single-point predictions,
    or even for something like seq2seq as long as the two
    sequences are the same length and s1[:t] informs s2[:t].
    When ordering and length aren't as clear-cut, use encoder-decoder.
    """

    X = tf.unstack(X, timesteps, 1)  # data, length of axis, axis -- tf rnn formatting
    cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(cell, X, dtype=tf.float32)  # hidden and cell states

    return tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])

"""graph!"""
with tf.name_scope("Model"):
    logits = RNN(X, weights, biases)
    predictions = tf.nn.softmax(logits)

"""loss"""
with tf.name_scope("Loss"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(predictions, axis=1)), tf.float32))

with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    opt_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()

"""summary"""
tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("learning_rate", lr)
merged_summary_op = tf.summary.merge_all()

"""training loop"""
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logdir=LOG_DIR, graph=tf.get_default_graph())

    for epoch in range(epochs):
        print("Epoch: {0}".format(epoch))

        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_inputs))  # take flattened 784 --> 28x28
        _, c, acc, summary = sess.run([opt_op, cost, accuracy, merged_summary_op],
                                      feed_dict={X: batch_x, y: batch_y, lr: learning_rate * (decay_rate ** epoch)})

        summary_writer.add_summary(summary, global_step=epoch)

        if epoch % 100 == 1:
            print("Epoch: {0}, Loss: {1}, Accuracy: {2}".format(epoch, c, acc))

    """test accuracy"""
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_inputs))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, y: test_label}))

utils.tensorboard_output(LOG_DIR)