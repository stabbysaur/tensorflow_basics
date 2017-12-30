import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

PATH = 'C:\\Users\\Stabby\\Documents\\Python\\Tensorflow\\'
LOG_DIR = PATH + 'logs\\logreg\\'
DATA_DIR = PATH + 'data\\'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

learning_rate = 0.002
decay_rate = 0.97
epochs = 25
batch_size = 100

"""inputs"""
input_data = tf.placeholder(tf.float32, shape=[None, 28*28])  # mnist 28x28
target = tf.placeholder(tf.float32, shape=[None, 10])  # 10 digit classes

"""learning rate with decay"""
lr = tf.Variable(initial_value=learning_rate,
                 trainable=False)

"""model weights"""
W = tf.Variable(tf.zeros([28*28, 10]), trainable=True)
b = tf.Variable(tf.zeros([10]), trainable=True)

with tf.name_scope('Model'):
    preds = tf.nn.softmax(tf.matmul(input_data, W) + b)

with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                   logits=(tf.matmul(input_data, W) + b)))
    correct_preds = tf.equal(tf.argmax(preds, 1),
                             tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))  # correct_preds has bools, must cast to float

with tf.name_scope('Optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

init = tf.global_variables_initializer()

"""tensorboard summaries"""
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

        num_batches = mnist.train.num_examples // batch_size
        for batch in range(num_batches):
            xs, ys = mnist.train.next_batch(batch_size)
            _, c, acc, summary = sess.run([optimizer, cost, accuracy, merged_summary_op],
                                     feed_dict={input_data: xs,
                                                target: ys,
                                                lr: learning_rate * (decay_rate ** epoch)})

            summary_writer.add_summary(summary, global_step=epoch*num_batches + batch)
            print("Cost: {0}, Accuracy: {1}".format(c, acc))

    print("Test accuracy: {0}".format(sess.run([accuracy], feed_dict={input_data: mnist.test.images,
                                                                      target: mnist.test.labels})))

"""output tensorboard commands"""
print("Run the command line:\n" \
      "--> tensorboard --logdir={0} " \
      "\nThen open localhost:6006/ into your web browser".format(LOG_DIR))