# BiasGradOp

import tensorflow as tf

data_format = "NHWC"
out_backprop = tf.constant(-0.151741251, shape=[5,2], dtype=tf.float32)
tf.raw_ops.BiasAddGrad(out_backprop=out_backprop, data_format=data_format)
