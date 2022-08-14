# InTopK

import tensorflow as tf

predictions = tf.constant(0.3, shape=[2,3], dtype=tf.float32)
targets = tf.constant([1,0], shape=[2], dtype=tf.int32)
k = tf.constant(3, shape=[], dtype=tf.int32)
tf.raw_ops.InTopK(predictions=predictions, targets=targets, k=k)
