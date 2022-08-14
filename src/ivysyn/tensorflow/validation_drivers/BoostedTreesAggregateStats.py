# BoostedTreesAggregateStatsOp

import tensorflow as tf

max_splits = 3
num_buckets = 4
node_ids = tf.constant(1, shape=[8], dtype=tf.int32)
gradients = tf.constant(.10, shape=[8,1], dtype=tf.float32)
hessians = tf.constant(.20, shape=[8,1], dtype=tf.float32)
feature = tf.constant(312, shape=[8,1], dtype=tf.int32)
tf.raw_ops.BoostedTreesAggregateStats(node_ids=node_ids, gradients=gradients, hessians=hessians, feature=feature, max_splits=max_splits, num_buckets=num_buckets)
