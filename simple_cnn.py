import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

import utils

PATH = 'C:\\Users\\Stabby\\Documents\\Python\\Tensorflow\\'
LOG_DIR = PATH + 'logs\\simple_cnn\\'
DATA_DIR = PATH + 'data\\'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

"""wipe out log_path"""
utils.wipe_dir(LOG_DIR)

learning_rate = 0.001
decay_rate = 1.00  # CAREFUL WITH DECAY RATE -- cross validate
epochs = 500
batch_size = 128

num_inputs = 28*28
num_classes = 10
dropout = 0.75

"""set up inputs"""
input_data = tf.placeholder(tf.float32, shape=[None, num_inputs])
target = tf.placeholder(tf.float32, shape=[None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout probability -- switch to 1.0 for testing
lr = tf.placeholder(tf.float32)

"""
set up filters and biases -- both conv and fully connected

input: 28x28x1
first conv layer: 5x5 filter x 32 channels, stride 1
--> 24x24x32, but with SAME padding input size is preserved
--> 28x28x32
first maxpool: 2x downsample
--> 14x14x32 
second conv layer: 5x5 filter x 64 channels, stride 1 
--> 14x14x32 (still with SAME padding) 
second maxpool: 2x downsample
--> 7x7x64
"""
filters = {'f1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
           'f2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
           'w1': tf.Variable(tf.random_normal([7*7*64, 1024])),
           'w2': tf.Variable(tf.random_normal([1024, 10]))}

"""biases match the output of each layer"""
biases = {'b1': tf.Variable(tf.random_normal([32])),
          'b2': tf.Variable(tf.random_normal([64])),
          'b3': tf.Variable(tf.random_normal([1024])),
          'b4': tf.Variable(tf.random_normal([10]))}

"""wrapper layer classes"""
def conv2d(X, F, b, stride=1):

    # F = filter. [filter_height, filter_width, in_channels, out_channels]
    X = tf.nn.conv2d(X, F, strides=[1, stride, stride, 1], padding='SAME')
    X = tf.nn.bias_add(X, b)
    return tf.nn.relu(X)

def maxpool2d(X, k=2):

    return tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def convnet(X, filters, biases, dropout):

    # reshape mnist data: [batch x dim x dim x channels]
    X = tf.reshape(X, shape=[-1, 28, 28, 1])

    # convolutional layers
    conv1 = conv2d(X, F=filters['f1'], b=biases['b1'], stride=1)
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, F=filters['f2'], b=biases['b2'], stride=1)
    conv2 = maxpool2d(conv2, k=2)

    # fully connected layers
    fc1 = tf.reshape(conv2, [-1, filters['w1'].get_shape().as_list()[0]])  # flatten
    fc1 = tf.add(tf.matmul(fc1, filters['w1']), biases['b3'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # output
    out = tf.add(tf.matmul(fc1, filters['w2']), biases['b4'])
    return out  # these are logits!! NO SOFTMAX ETC

with tf.name_scope("Model"):
    logits = convnet(input_data, filters, biases, keep_prob)
    predictions = tf.nn.softmax(logits)
with tf.name_scope("Loss"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                   logits=logits))
    correct_preds = tf.equal(tf.argmax(predictions, 1), tf.argmax(target, 1))
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
                                 feed_dict={input_data: batch_X,
                                            target: batch_y,
                                            lr: learning_rate * (decay_rate ** epoch),
                                            keep_prob: dropout})

        summary_writer.add_summary(summary, global_step=epoch)
        print("Cost: {0}, Accuracy: {1}".format(c, acc))

    print("Testing accuracy: {0}".format(sess.run(accuracy, feed_dict={input_data: mnist.test.images,
                                                                       target: mnist.test.labels,
                                                                       keep_prob: 1.0})))

utils.tensorboard_output(LOG_DIR)