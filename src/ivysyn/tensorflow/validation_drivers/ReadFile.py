# ReadFileOp

import tensorflow as tf

filename = tf.constant("/tmp/checkpoint_testd57rll0s/tmpm5mr4r49/iterator", shape=[], dtype=tf.string)
tf.raw_ops.ReadFile(filename=filename)
