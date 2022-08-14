# CompressElementOp

import tensorflow as tf

components = [tf.constant(1, shape=[], dtype=tf.int64)]
tf.raw_ops.CompressElement(components=components)
