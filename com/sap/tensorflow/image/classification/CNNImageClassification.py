import tensorflow as tf
from com.sap.tensorflow.image.classification.FruitsImageClassifier import FruitsImageClassifier
import argparse
import sys

class CNNImageClassification:
    def __init__(self):
        self.data = {}
        
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    def main(self, filePath):
        # Import data
        imageClassifier = FruitsImageClassifier()
        imageDataSets = imageClassifier.preProcessImages(filePath);
        
        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])
        
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])
        
        '''First Convolution Layer'''
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        x_image = tf.reshape(x, [-1,28,28,1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
    
        '''Second Convolution Layer'''
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
    
        '''Densly connected layer'''    
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        '''Dropout'''
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        '''Readout Layer'''
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
        '''
        The raw formulation of cross-entropy,
        
        tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        
        can be numerically unstable. So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
        outputs of 'y', and then average across the batch.
        '''
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        sess = tf.Session()    
        sess.run(tf.global_variables_initializer())

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        
        for i in range(1500):
            batch = imageClassifier.getNextTrainingBatch(75)
            if i%100 == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                '''
                # Update the events file.
                summary_str = sess.run(summary, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                summary_writer.add_summary(summary_str, i)
                summary_writer.flush()
                '''
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={x: imageDataSets['test'][0], y_: imageDataSets['test'][1], keep_prob: 1.0}))


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    filePath = 'C:\\D\\Fruits_Dataset\\FIDS30\\'
    cnnClassifier = CNNImageClassification()
    cnnClassifier.main(filePath)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/fruits/logs/fruits_image_classifier', help='Directory to put the log data.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)