# BoostedTreesSparseAggregateStatsOp

import tensorflow as tf

max_splits = 2
num_buckets = 2
node_ids = tf.constant(0, shape=[10], dtype=tf.int32)
gradients = tf.constant(2.41642356, shape=[10,2], dtype=tf.float32)
hessians = tf.constant(2.44031596, shape=[10,2], dtype=tf.float32)
feature_indices = tf.constant(0, shape=[5,2], dtype=tf.int32)
feature_values = tf.constant(1, shape=[5], dtype=tf.int32)
feature_shape = tf.constant([10,1], shape=[2], dtype=tf.int32)
tf.raw_ops.BoostedTreesSparseAggregateStats(node_ids=node_ids, gradients=gradients, hessians=hessians, feature_indices=feature_indices, feature_values=feature_values, feature_shape=feature_shape, max_splits=max_splits, num_buckets=num_buckets)
