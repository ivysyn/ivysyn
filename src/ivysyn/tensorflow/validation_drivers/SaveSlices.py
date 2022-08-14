# SaveSlicesOp

import tensorflow as tf

filename = tf.constant("/tmp/saver_testnix_3upa/tmpqzvm40r3/ckpt_for_debug_string1", shape=[], dtype=tf.string)
tensor_names = tf.constant("[v0,v1]", shape=[2], dtype=tf.string)
shapes_and_slices = tf.constant("[]", shape=[2], dtype=tf.string)
data = tf.constant(1, shape=[2,3], dtype=tf.float32)
tf.raw_ops.SaveSlices(filename=filename, tensor_names=tensor_names, shapes_and_slices=shapes_and_slices, data=data)
