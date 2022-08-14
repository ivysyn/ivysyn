# BucketizeOp

import tensorflow as tf

boundaries = [5, 10, 15]
input = tf.constant(1315, shape=[2,1], dtype=tf.float32)
tf.raw_ops.Bucketize(input=input, boundaries=boundaries)
