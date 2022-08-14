# WriteFileOp

import tensorflow as tf

filename = tf.constant("/tmp/checkpoint_testd57rll0s/tmpm5mr4r49/iterator", shape=[], dtype=tf.string)
contents = tf.constant("\010\025\022\004\022\002\010\001\"ts\n\024tensorflow", shape=[], dtype=tf.string)
tf.raw_ops.WriteFile(filename=filename, contents=contents)
