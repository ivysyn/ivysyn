# SparseCrossV2Op

import tensorflow as tf

indices = [tf.constant([], shape=[0, 2], dtype=tf.int64), tf.constant(
    [], shape=[0, 2], dtype=tf.int64), tf.constant([], shape=[0, 2], dtype=tf.int64)]
values = [tf.constant("", shape=[0], dtype=tf.string), tf.constant(
    "", shape=[0], dtype=tf.string), tf.constant("", shape=[0], dtype=tf.string)]
shapes = [tf.constant(0, shape=[2], dtype=tf.int64), tf.constant(
    0, shape=[2], dtype=tf.int64), tf.constant(0, shape=[2], dtype=tf.int64)]
dense_inputs = []
sep = tf.constant("_X_", shape=[], dtype=tf.string)
tf.raw_ops.SparseCrossV2(indices=indices, values=values,
                         shapes=shapes, dense_inputs=dense_inputs, sep=sep)
