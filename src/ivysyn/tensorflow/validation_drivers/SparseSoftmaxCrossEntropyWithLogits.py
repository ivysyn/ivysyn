# SparseSoftmaxXentWithLogitsOp

import tensorflow as tf

features = tf.constant(0.274831653, shape=[6,4], dtype=tf.float32)
labels = tf.constant(1, shape=[6], dtype=tf.int32)
tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits(features=features, labels=labels)
