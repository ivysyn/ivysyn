# SparseSplitOp

import tensorflow as tf

num_split = 3
split_dim = tf.constant(1, shape=[], dtype=tf.int64)
indices = tf.constant(0, shape=[5,2], dtype=tf.int64)
values = tf.constant(2, shape=[5], dtype=tf.int32)
shape = tf.constant([2,7], shape=[2], dtype=tf.int64)
tf.raw_ops.SparseSplit(split_dim=split_dim, indices=indices, values=values, shape=shape, num_split=num_split)
