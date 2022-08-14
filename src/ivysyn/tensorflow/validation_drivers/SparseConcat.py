# SparseConcatOp

import tensorflow as tf

concat_dim = -1
indices = [tf.constant(0, shape=[4, 2], dtype=tf.int64),
           tf.constant(1, shape=[4, 2], dtype=tf.int64)]
values = [tf.constant(1, shape=[4], dtype=tf.float32),
          tf.constant(1, shape=[4], dtype=tf.float32)]
shapes = [tf.constant(3, shape=[2], dtype=tf.int64),
          tf.constant(3, shape=[2], dtype=tf.int64)]
tf.raw_ops.SparseConcat(indices=indices, values=values,
                        shapes=shapes, concat_dim=concat_dim)
