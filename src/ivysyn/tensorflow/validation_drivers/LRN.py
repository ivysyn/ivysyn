# LRNOp

import tensorflow as tf

depth_radius = 1
bias = 1.00591636
alpha = 0.322271705
beta = 0.687533557
input = tf.constant(0.472314954, shape=[12,13,1,14], dtype=tf.float32)
tf.raw_ops.LRN(input=input, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)
