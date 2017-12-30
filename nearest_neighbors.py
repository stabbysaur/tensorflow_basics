import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PATH = 'C:\\Users\\Stabby\\Documents\\Python\\Tensorflow\\'
LOG_DIR = PATH + 'logs\\knn\\'
DATA_DIR = PATH + 'data\\'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

"""reduce dataset size for this example"""
X_train, y_train = mnist.train.next_batch(5000)
X_test, y_test = mnist.test.next_batch(200)

"""inputs -- 1NN only"""
train = tf.placeholder(tf.float32, [None, 28*28])  # placeholder for the nearest neighbor
test = tf.placeholder(tf.float32, [28*28])  # sample we're trying to classify

"""distances"""
distances = tf.reduce_sum(tf.pow(train - test, 2), reduction_indices=1)  # distances by rows
pred = tf.argmin(distances, 0)

init = tf.global_variables_initializer()

"""run session"""
acc = 0.0
with tf.Session() as sess:
    sess.run(init)

    for i in range(200):
        nn_index = sess.run(pred, feed_dict={train: X_train,
                                             test: X_test[i,:]})
        print("Test: {0}, Prediction: {1}, True: {2}".format(i, np.argmax(y_train[nn_index]),
                                                             np.argmax(y_test[i])))
        if np.argmax(y_train[nn_index]) == np.argmax(y_test[i]):
            acc += 1. / len(X_test)

    print("Done! Accuracy: {0}".format(acc))