from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from com.sap.tensorflow.image.classification.FruitsImageClassifier import FruitsImageClassifier
import argparse
import sys
from tensorflow.contrib.tensorboard.plugins import projector
import os

class CNNImageClassification:
    def __init__(self):
        self.data = {}
        
    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def feed_dict(self, train, batch, x, y_, keep_prob):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        k = 0
        if train:
            k = FLAGS.dropout
        else:
            k = 1.0
        return {x: batch[0], y_: batch[1], keep_prob: k}    
    
    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    
    def main(self, filePath):
        # Import data
        imageClassifier = FruitsImageClassifier()
        imageDataSets = imageClassifier.preProcessImages(filePath);
        
        # Create the model
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        with tf.name_scope('input_reshape'):
            x_image = tf.reshape(x, [-1,28,28,1])
            tf.summary.image('input', x_image, 10)

        '''First Convolution Layer'''
        with tf.name_scope('First_Conv_Layer'):
            with tf.name_scope('weights'):
                W_conv1 = self.weight_variable([5, 5, 1, 32], 'weights-layer1')
                self.variable_summaries(W_conv1)
                tf.summary.histogram('weights-layer1', W_conv1)
        with tf.name_scope('biases'):
            b_conv1 = self.bias_variable([32], 'bias-layer1')
            self.variable_summaries(b_conv1)
            tf.summary.histogram('bias-layer1', b_conv1)
        with tf.name_scope('Weigths_Plus_bias'):
            preactivate = self.conv2d(x_image, W_conv1) + b_conv1
            h_conv1 = tf.nn.relu(preactivate)
            tf.summary.histogram('pre_activations', h_conv1)                        
            h_pool1 = self.max_pool_2x2(h_conv1)
    
        '''Second Convolution Layer'''
        with tf.name_scope('Second_Conv_Layer'):
            with tf.name_scope('weights'):
                W_conv2 = self.weight_variable([5, 5, 32, 64], 'weights-layer2')
                self.variable_summaries(W_conv2)
                tf.summary.histogram('weights-layer2', W_conv2)                                                
        with tf.name_scope('biases'):
            b_conv2 = self.bias_variable([64], 'bias-layer2')
            self.variable_summaries(b_conv2)
            tf.summary.histogram('bias-layer2', b_conv2)                                                
        with tf.name_scope('Weigths_Plus_bias'):
            preactivate = self.conv2d(h_pool1, W_conv2) + b_conv2
            h_conv2 = tf.nn.relu(preactivate)
            tf.summary.histogram('pre_activations', h_conv2)                                    
            h_pool2 = self.max_pool_2x2(h_conv2)
    
        '''Densly connected layer'''
        with tf.name_scope('Densly_connected_Layer'):
            with tf.name_scope('weights'):
                W_fc1 = self.weight_variable([7 * 7 * 64, 1024], 'weights-FC')
                self.variable_summaries(W_fc1)
                tf.summary.histogram('weights', W_fc1)
        with tf.name_scope('biases'):
            b_fc1 = self.bias_variable([1024], 'bias-FC')
            self.variable_summaries(b_fc1)
            tf.summary.histogram('bias-FC', b_fc1)                        
        with tf.name_scope('Weigths_Plus_bias'):
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            preactivate = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
            h_fc1 = tf.nn.relu(preactivate)
            tf.summary.histogram('pre_activations', h_fc1)            
        
        '''Dropout'''
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32, name='dropout')
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        '''Readout Layer'''
        W_fc2 = self.weight_variable([1024, 10], 'weights-output')
        b_fc2 = self.bias_variable([10], 'bias-output')
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
        '''
        The raw formulation of cross-entropy,
        
        tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        
        can be numerically unstable. So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
        outputs of 'y', and then average across the batch.
        '''
        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)
        
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        # Build the summary Tensor based on the TF collection of Summaries.
        merged = tf.summary.merge_all()
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        sess = tf.Session()    
        # Instantiate a SummaryWriter to output summaries and the Graph.
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)        
        sess.run(tf.global_variables_initializer())
        # Create randomly initialized embedding weights which will be trained.
        N = 10000 # Number of items (vocab size).
        D = 200 # Dimensionality of the embedding.
        embedding_var = tf.Variable(tf.random_normal([N,D]), name='image_embedding')
        
        # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
        config = projector.ProjectorConfig()
        
        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        #embedding.sprite.image_path = os.path.join(FLAGS.log_dir, 'sprite.png')
        embedding.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
        #embedding.sprite.single_image_dim.extend([28,28])
        
        # Use the same LOG_DIR where you stored your checkpoint.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
        
        # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
        # read this file during startup.
        projector.visualize_embeddings(summary_writer, config)        
        
        for i in range(FLAGS.max_steps):
            batch = imageClassifier.getNextTrainingBatch(FLAGS.training_batch)
            if i % 100 == 0:  # Record summaries and test-set accuracy
                summary, acc = sess.run([merged, accuracy], feed_dict=self.feed_dict(False, batch, x, y_, keep_prob))
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, acc))
            else: 
                if i % 500 == 499:
                    #train_accuracy = accuracy.eval(session=sess, feed_dict=self.feed_dict(False, batch, x, y_, keep_prob))
                    train_accuracy = accuracy.eval(session=sess, feed_dict=self.feed_dict(False, batch, x, y_, keep_prob))
                    print("step %d, training accuracy %g"%(i, train_accuracy))
                    
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                  feed_dict=self.feed_dict(True, batch, x, y_, keep_prob),
                                  options=run_options,
                                  run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)
                if i%1000 == 999:
                    sess.run([merged, train_step], feed_dict=self.feed_dict(False, imageDataSets['test'], x, y_, keep_prob))
                    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')         
                    saver.save(sess, checkpoint_file,  global_step=i)
                    print('Adding run embeddings for', i)
                else: # Record a summary
                    #train_step.run(session=sess, feed_dict=self.feed_dict(True, batch, x, y_, keep_prob))
                    summary, _ = sess.run([merged, train_step], feed_dict=self.feed_dict(True, batch, x, y_, keep_prob))
                    train_writer.add_summary(summary, i)
        test_accuracy = accuracy.eval(session=sess, feed_dict=self.feed_dict(False, imageDataSets['test'], x, y_, keep_prob))
        print("test accuracy %g"%test_accuracy)
        train_writer.close()
        test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    #filePath = 'C:\\Supermarket_Produce_Dataset\\Fruits'
    cnnClassifier = CNNImageClassification()
    cnnClassifier.main(FLAGS.image_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='C:\\Supermarket_Produce_Dataset\\Fruits', help='Image path.')    
    parser.add_argument('--max_steps', type=int, default=10000, help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--training_batch', type=int, default=23, help='Successive batch rate')    
    parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout.')    
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/fruits/logs/fruits_image_classifier', help='Directory to put the log data.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)