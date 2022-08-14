# HSVToRGBOp

import tensorflow as tf

images = tf.constant(0.41910547, shape=[5,2,7,3], dtype=tf.float32)
tf.raw_ops.HSVToRGB(images=images)
