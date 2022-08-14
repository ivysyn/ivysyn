# LRNGradOp

import tensorflow as tf

depth_radius = 1
bias = 1.59018219
alpha = 0.117728651
beta = 0.404427052
input_grads = tf.constant(1, shape=[4,1,1,2], dtype=tf.float32)
input_image = tf.constant(0.542408228, shape=[4,1,1,2], dtype=tf.float32)
output_image = tf.constant(0.445362538, shape=[4,1,1,2], dtype=tf.float32)
tf.raw_ops.LRNGrad(input_grads=input_grads, input_image=input_image, output_image=output_image, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)
