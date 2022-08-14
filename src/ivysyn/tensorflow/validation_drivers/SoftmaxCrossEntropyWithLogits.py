# SoftmaxXentWithLogitsOp

import tensorflow as tf

features = tf.constant(0.0368289575, shape=[32,10], dtype=tf.float32)
labels = tf.constant(0.347844362, shape=[32,10], dtype=tf.float32)
tf.raw_ops.SoftmaxCrossEntropyWithLogits(features=features, labels=labels)
