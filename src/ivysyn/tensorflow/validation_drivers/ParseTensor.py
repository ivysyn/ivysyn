# ParseTensorOp

import tensorflow as tf

out_type = tf.variant
serialized = tf.constant("\010\025\022\004\022\002\010\001\"ts\n\024tensorflow", shape=[], dtype=tf.string)
tf.raw_ops.ParseTensor(serialized=serialized, out_type=out_type)
