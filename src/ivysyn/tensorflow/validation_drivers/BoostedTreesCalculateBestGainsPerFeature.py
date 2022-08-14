# BoostedTreesCalculateBestGainsPerFeatureOp

import tensorflow as tf

max_splits = 7
node_id_range = tf.constant([1,3], shape=[2], dtype=tf.int32)
stats_summary_list = tf.constant(0, shape=[7,4,2], dtype=tf.float32)
l1 = tf.constant(0, shape=[7,4,2], dtype=tf.float32)
l2 = tf.constant(0.1, shape=[], dtype=tf.float32)
tree_complexity = tf.constant(0, shape=[], dtype=tf.float32)
min_node_weight = tf.constant(0, shape=[], dtype=tf.float32)
tf.raw_ops.BoostedTreesCalculateBestGainsPerFeature(node_id_range=node_id_range, stats_summary_list=stats_summary_list, l1=l1, l2=l2, tree_complexity=tree_complexity, min_node_weight=min_node_weight, max_splits=max_splits)
