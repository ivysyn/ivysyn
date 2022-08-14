# OptionalFromValueOp

import tensorflow as tf

components = tf.constant([0,1], shape=[2], dtype=tf.int32)
tf.raw_ops.OptionalFromValue(components=components)
