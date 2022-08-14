# UnravelIndexOp

import tensorflow as tf

indices = tf.constant([2,5,7], shape=[3], dtype=tf.int32)
dims = tf.constant([3,0], shape=[2], dtype=tf.int32)
tf.raw_ops.UnravelIndex(indices=indices, dims=dims)
