# StringToKeyedHashBucketOp

import tensorflow as tf

num_buckets = 1000
key = [1231, 12512]
input = tf.constant("[abcd,efgh,aabbccdd]", shape=[3], dtype=tf.string)
tf.raw_ops.StringToHashBucketStrong(input=input, num_buckets=num_buckets, key=key)
