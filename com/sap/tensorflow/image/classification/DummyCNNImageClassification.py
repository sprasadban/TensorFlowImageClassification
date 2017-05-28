"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from com.sap.tensorflow.image.classification.FruitsImageClassifier import FruitsImageClassifier
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main(_):
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    imageClassifier = FruitsImageClassifier(FLAGS.image_count)
    #imageDataSets = imageClassifier.preProcessRandomImages(FLAGS.data_dir);
    filters = FLAGS.filter_file.split()    
    imageDataSets = imageClassifier.loadPreprocessedImages(FLAGS.data_dir, filters)
    print('Training length=',len(imageDataSets['training'][0]))
    print('Test length=',len(imageDataSets['test'][0]))
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    '''First Convolution Layer'''
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    '''Second Convolution Layer'''
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    '''Densly connected layer'''    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    '''Dropout'''
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    '''Readout Layer'''
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    '''
    The raw formulation of cross-entropy,
    
    tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    
    can be numerically unstable. So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    outputs of 'y', and then average across the batch.
    '''
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.Session()    
    sess.run(tf.global_variables_initializer())
    accuracyCount = 0
    for i in range(FLAGS.max_steps):
        #batch = mnist.train.next_batch(50)
        batch = imageClassifier.getNextTrainingBatch(FLAGS.training_batch)
        #batch = imageDataSets['training']
        if i%100 == 0:
            train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            if(train_accuracy > 0.99):
                accuracyCount = accuracyCount + 1
            else:
                accuracyCount = 0
            if(accuracyCount == 10):
                break
        train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        
    #print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={x: imageDataSets['test'][0], y_: imageDataSets['test'][1], keep_prob: 1.0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''C:\\Supermarket_Produce_Dataset\\Fruits'''
    parser.add_argument('--data_dir', type=str, default='C:\\Supermarket_Produce_Dataset\\Fruits', help='Directory for storing input data')
    parser.add_argument('--filter_file', type=str, default='_seg_com, _com', help='Directory for storing input data')    
    parser.add_argument('--max_steps', type=int, default=15000, help='Number of steps to run trainer.')
    parser.add_argument('--image_count', type=int, default=1671, help='Number of images to run trainer.')    
    parser.add_argument('--training_batch', type=int, default=50, help='Successive batch rate') 
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')       
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)