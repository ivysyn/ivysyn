# SparseCrossHashedOp

import tensorflow as tf

indices = [tf.constant(0, shape=[3, 2], dtype=tf.int64), tf.constant(
    0, shape=[1, 2], dtype=tf.int64), tf.constant(0, shape=[2, 2], dtype=tf.int64)]
values = [tf.constant("[batch1-FC1-F1,batch1-FC1-F2,batch1-FC1-F3]", shape=[3], dtype=tf.string), tf.constant(
    "-FC2", shape=[1], dtype=tf.string), tf.constant("[batch1-FC3-F1,batch1-FC3-F2]", shape=[2], dtype=tf.string)]
shapes = [tf.constant([1, 3], shape=[2], dtype=tf.int64), tf.constant(
    [1, 3], shape=[2], dtype=tf.int64), tf.constant([1, 3], shape=[2], dtype=tf.int64)]
dense_inputs = []
num_buckets = tf.constant(1000, shape=[], dtype=tf.int64)
strong_hash = tf.constant(False, shape=[], dtype=tf.bool)
salt = tf.constant([137, 173], shape=[2], dtype=tf.int64)
tf.raw_ops.SparseCrossHashed(indices=indices, values=values, shapes=shapes,
                             dense_inputs=dense_inputs, num_buckets=num_buckets, strong_hash=strong_hash, salt=salt)
