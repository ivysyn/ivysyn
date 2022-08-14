# BoostedTreesSparseCalculateBestFeatureSplitOp

import tensorflow as tf

logits_dimension = 1
split_type = "inequality"
node_id_range = tf.constant([1,3], shape=[2], dtype=tf.int32)
stats_summary_indices = tf.constant(1, shape=[24,4], dtype=tf.int32)
stats_summary_values = tf.constant(0.15, shape=[24], dtype=tf.float32)
stats_summary_shape = tf.constant(7, shape=[4], dtype=tf.int32)
l1 = tf.constant(0, shape=[], dtype=tf.float32)
l2 = tf.constant(0, shape=[], dtype=tf.float32)
tree_complexity = tf.constant(0, shape=[], dtype=tf.float32)
min_node_weight = tf.constant(0, shape=[], dtype=tf.float32)
tf.raw_ops.BoostedTreesSparseCalculateBestFeatureSplit(node_id_range=node_id_range, stats_summary_indices=stats_summary_indices, stats_summary_values=stats_summary_values, stats_summary_shape=stats_summary_shape, l1=l1, l2=l2, tree_complexity=tree_complexity, min_node_weight=min_node_weight, logits_dimension=logits_dimension, split_type=split_type)
