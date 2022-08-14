# BoostedTreesBucketizeOp

import tensorflow as tf

float_values = tf.constant(1.2, shape=[6], dtype=tf.float32)
bucket_boundaries = tf.constant(2.3, shape=[6], dtype=tf.float32)
tf.raw_ops.BoostedTreesBucketize(float_values=float_values, bucket_boundaries=bucket_boundaries)
