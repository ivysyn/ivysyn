# ReverseV2Op

import tensorflow as tf

tensor = tf.constant(0.184634328, shape=[2,5,8,3], dtype=tf.float32)
axis = tf.constant(2, shape=[1], dtype=tf.int32)
tf.raw_ops.ReverseV2(tensor=tensor, axis=axis)
