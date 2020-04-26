""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time
import utils

# Define paramaters for the model
learning_rate = 0.0001
batch_size = 8
n_epochs = 10
n_train = 60000
n_test = 10000

mnist_folder = 'data/mnist'
if os.path.isdir(mnist_folder) != True:
    os.mkdir('data')
    os.mkdir(mnist_folder)
utils.download_mnist(mnist_folder)
print(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)
train_data = train_data.batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
img, label = iterator.get_next()
img = tf.reshape(img, [-1, 28, 28, 1])
train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

is_train = tf.placeholder(tf.bool, shape=())


def conv2d(x, w, b, layer, is_train):
    with tf.variable_scope(layer, reuse=False) as scope:
        x = batch_norm_wrapper(x, layer, is_train)
        x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        x = tf.nn.relu(x)
        return x


def batch_norm_wrapper(x, layer, is_train):
    '''TODO: batch normalization'''
    with tf.variable_scope('batch_norm_{}'.format(layer)) as scope:
        test = tf.layers.batch_normalization(x)
        train = x
        x = tf.cond(tf.equal(is_train, tf.constant(True)), lambda: test, lambda: train)
        return x


def cnn_graph():

    with tf.variable_scope('ConvNet', reuse=False) as scope:
        w = {
            # Convolution Layers
            'c1': tf.get_variable('weight_c1', shape=(5,5,1,16), \
                    initializer=tf.contrib.layers.xavier_initializer()), 
            'c2': tf.get_variable('weight_c2', shape=(5,5,16,32), \
                    initializer=tf.contrib.layers.xavier_initializer()), #16

            'fc1': tf.get_variable('weight_fc1', shape=(7*7*128,256), 
                    initializer=tf.contrib.layers.xavier_initializer()),
            'fc2': tf.get_variable('weight_fc2', shape=(256,10), 
                    initializer=tf.contrib.layers.xavier_initializer()),
        }
        b = {
            'c1': tf.get_variable('bias_c1', shape=(16), initializer=tf.zeros_initializer()),
            'c2': tf.get_variable('bias_c2', shape=(32), initializer=tf.zeros_initializer()), #16

            'fc1': tf.get_variable('bias_fc1', shape=(256), initializer=tf.zeros_initializer()),
            'fc2': tf.get_variable('bias_fc2', shape=(10), initializer=tf.zeros_initializer()),
        }

        conv1 = conv2d(img, w['c1'], b['c1'], 'conv1', is_train)
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        conv2 = conv2d(conv1, w['c2'], b['c2'], 'conv2', is_train)
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        flatten = tf.reshape(pool2, [-1, w['fc1'].get_shape().as_list()[0]]) 

        with tf.variable_scope('dense_layer', reuse=False) as scope:
            fc_layer1 = tf.add(tf.matmul(flatten, w['fc1']), b['fc1']) 
            fc_layer1 = batch_norm_wrapper(fc_layer1, 'pool2', is_train)
            fc_layer1 = tf.nn.relu(fc_layer1)
            _test = tf.nn.dropout(fc_layer1, 0.5)
            _train = fc_layer1
            fc_layer1 = tf.cond(tf.equal(is_train, tf.constant(True)), lambda: _test, lambda: _train)

        with tf.variable_scope('output_layer', reuse=False) as scope:
            logits = tf.add(tf.matmul(fc_layer1, w['fc2']), b['fc2']) 
        prediction = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
        loss = tf.reduce_mean(entropy, name='loss')
        optimizer = tf.contrib.opt.NadamOptimizer(learning_rate).minimize(loss)
        preds = tf.nn.softmax(logits)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

        return prediction, loss, optimizer, accuracy



prediction, loss, optimizer, accuracy = cnn_graph()

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    #training
    for i in range(n_epochs):   
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss], feed_dict={is_train: True})
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i+1, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    #testing
    sess.run(test_init)
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run([accuracy], feed_dict={is_train: False})
            total_correct_preds += accuracy_batch[0]
    except tf.errors.OutOfRangeError:
        pass
    print('Accuracy {0}'.format(total_correct_preds/n_test))
writer.close()
