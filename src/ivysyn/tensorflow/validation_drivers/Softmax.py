# SoftmaxOp

import tensorflow as tf

logits = tf.constant(-1.73661923, shape=[100,2], dtype=tf.float32)
tf.raw_ops.Softmax(logits=logits)
