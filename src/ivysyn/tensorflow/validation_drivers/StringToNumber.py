# StringToNumberOp

import tensorflow as tf

out_type = tf.float32
string_tensor = tf.constant("[-2.0,3.0,-3.0]", shape=[3], dtype=tf.string)
tf.raw_ops.StringToNumber(string_tensor=string_tensor, out_type=out_type)
