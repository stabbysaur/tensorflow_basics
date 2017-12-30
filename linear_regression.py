import tensorflow as tf
import numpy
import numpy.random as rand
import matplotlib.pyplot as plt
import seaborn as sns

LOG_DIR = 'C:\\Users\\Stabby\\Documents\\Python\\Tensorflow\\logs\\linreg\\'

learning_rate = 0.001
epochs = 1000

"""input data"""
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

"""tf graph"""
input_data = tf.placeholder(tf.float32)
target = tf.placeholder(tf.float32)

W = tf.Variable(initial_value=rand.normal(), name='weight', trainable=True)
b = tf.Variable(initial_value=rand.normal(), name='bias', trainable=True)

"""use name scopes for tensorboard"""
with tf.name_scope("Model"):
    pred = tf.add(tf.multiply(W, input_data), b)

with tf.name_scope("Loss"):
    cost = tf.reduce_mean(tf.pow(pred - target, 2)) / n_samples

with tf.name_scope("Optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

"""set up summaries that should be tracked through tensorboard"""
tf.summary.scalar("loss", cost)
merged_summary_op = tf.summary.merge_all()  # combine summaries into single op

"""training loop"""
with tf.Session() as sess:

    sess.run(init)  # initialize variables
    summary_writer = tf.summary.FileWriter(logdir=LOG_DIR, graph=tf.get_default_graph())

    for epoch in range(epochs):
        epoch_cost = 0.0
        print("Epoch: {0}".format(epoch))

        for x, y in zip(train_X, train_Y):
            _, temp_cost, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={input_data:x,
                                                                                               target:y})

            summary_writer.add_summary(summary, epoch)
            epoch_cost += temp_cost

        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(epoch_cost),
              "W=", sess.run(W), "b=", sess.run(b))

    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

"""output tensorboard commands"""
print("Run the command line:\n" \
      "--> tensorboard --logdir={0} " \
      "\nThen open localhost:6006/ into your web browser".format(LOG_DIR))