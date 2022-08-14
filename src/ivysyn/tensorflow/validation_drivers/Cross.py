# CrossOp

import tensorflow as tf

a = tf.constant(0.455004334, shape=[4,2,3], dtype=tf.float32)
b = tf.constant(0.448441505, shape=[4,2,3], dtype=tf.float32)
tf.raw_ops.Cross(a=a, b=b)
