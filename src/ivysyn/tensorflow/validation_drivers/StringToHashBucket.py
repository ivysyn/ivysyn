# LegacyStringToHashBucketOp

import tensorflow as tf

num_buckets = 1000
string_tensor = tf.constant("[abcd,efgh,aabbccdd]", shape=[3], dtype=tf.string)
tf.raw_ops.StringToHashBucket(string_tensor=string_tensor, num_buckets=num_buckets)
