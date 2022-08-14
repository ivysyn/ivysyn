# DynamicPartitionOp

import tensorflow as tf

num_partitions = 2
data = tf.constant(0, shape=[5], dtype=tf.int32)
partitions = tf.constant(1, shape=[5], dtype=tf.int32)
tf.raw_ops.DynamicPartition(data=data, partitions=partitions, num_partitions=num_partitions)
