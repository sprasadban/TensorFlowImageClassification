'''
Created on May 6, 2017

@author: I050385
'''

from numpy import core
import tensorflow as tf

def addition(x, y):
    return core.absolute(x+y)

def testTensorflow():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
    
if __name__ == '__main__':
    print(addition(10, 20))
    testTensorflow()
