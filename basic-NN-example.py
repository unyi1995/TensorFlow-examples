import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# parameters
epochs = 10
batch_size = 100
# size of hidden layer
h = 392

# MNIST database
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train, y_train = mnist.train.images, mnist.train.labels
x_validation, y_validation = mnist.validation.images, mnist.validation.labels
x_test, y_test = mnist.test.images, mnist.test.labels

# placeholders
n_features = x_train.shape[1]
x = tf.placeholder(tf.float32, shape=(None, n_features), name='x')
n_labels = y_train.shape[1]
y = tf.placeholder(tf.float32, shape=(None, n_labels), name='y')

# layer1
W1 = tf.Variable(tf.random_normal((n_features, h), stddev=0.01), name='W1')
b1 = tf.Variable(tf.random_normal((h,)), name='b1')
x1 = tf.matmul(x, W1) + b1
layer1 = tf.nn.relu(x1, name='layer1')

# layer2
W2 = tf.Variable(tf.random_normal((h, n_labels), stddev=0.01), name='W2')
b2 = tf.Variable(tf.random_normal((n_labels,)), name='b2')
x2 = tf.matmul(layer1, W2) + b2
layer2 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x2, name='layer2')

# optimizer
loss = tf.reduce_mean(layer2, name='loss')
optimizer = tf.train.AdamOptimizer().minimize(loss)

# accuracy
success = tf.equal(tf.argmax(x2, 1), tf.argmax(y, 1), name='success')
accuracy = tf.reduce_mean(tf.cast(success, tf.float32), name='accuracy')

initializer = tf.global_variables_initializer()

session = tf.InteractiveSession()

session.run(initializer)

for epoch in range(0, epochs):

  # training on batches
  for i in range(0, len(y_train) // batch_size):
    start = i * batch_size
    end = (i + 1) * batch_size
    x_train_batch, y_train_batch = x_train[start:end], y_train[start:end]
    feed_dict_train_batch = {x: x_train_batch, y: y_train_batch}
    session.run(optimizer, feed_dict=feed_dict_train_batch)

  # validation
  feed_dict_validation = {x: x_validation, y: y_validation}
  loss_result, accuracy_result = session.run([loss, accuracy], feed_dict=feed_dict_validation)
  print("Epoch: " + str(epoch) + " " + "Validation loss: " + str(loss_result) + " " + "Validation accuracy: " + str(accuracy_result))

# test
feed_dict_test = {x: x_test, y: y_test}
loss_result, accuracy_result = session.run([loss, accuracy], feed_dict=feed_dict_test)
print("Test loss: " + str(loss_result) + " " + "Test accuracy: " + str(accuracy_result))

session.close()
