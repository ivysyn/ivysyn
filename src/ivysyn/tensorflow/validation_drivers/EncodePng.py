# EncodePngOp

import tensorflow as tf

compression = 7
image = tf.constant(0, shape=[200,256,3], dtype=tf.uint8)
tf.raw_ops.EncodePng(image=image, compression=compression)
