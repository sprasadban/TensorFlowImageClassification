import tensorflow as tf

def testTensorflow():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0, tf.float32)
    print(node1, node2)
    sess = tf.Session()
    print(sess.run([node1, node2]))
    node3 = tf.add(node1, node2)
    print(node3)
    print(sess.run(node3))

def testTensorflowPlaceholders():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    ''' Same as tf.add(a, b)'''
    adder_node = a + b
    add_and_tripple = adder_node * 3
    sess = tf.Session()    
    print(sess.run(add_and_tripple, {a:3, b:4.5}))
    print(sess.run(add_and_tripple, {a:[1,3], b:[4.5,5.5]}))
    
if __name__ == '__main__':
    testTensorflow()
    testTensorflowPlaceholders()