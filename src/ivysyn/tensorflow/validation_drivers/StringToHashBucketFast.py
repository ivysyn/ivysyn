# StringToHashBucketOp

import tensorflow as tf

num_buckets = 1000
input = tf.constant("[abcd,efgh,aabbccdd]", shape=[3], dtype=tf.string)
tf.raw_ops.StringToHashBucketFast(input=input, num_buckets=num_buckets)
