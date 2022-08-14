# PackOp

import tensorflow as tf

axis = 0
values = [tf.constant("-00000", shape=[], dtype=tf.string)]
tf.raw_ops.Pack(values=values, axis=axis)
