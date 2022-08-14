# SaveV2

import tensorflow as tf

prefix = tf.constant("-00000", shape=[], dtype=tf.string)
tensor_names = [tf.constant("[Variable,Variable_1]",
                            shape=[2], dtype=tf.string)]
shape_and_slices = [tf.constant("[]", shape=[2], dtype=tf.string)]
tensors = [tf.constant(2, shape=[], dtype=tf.float32)]
tf.raw_ops.SaveV2(prefix=prefix, tensor_names=tensor_names,
                  shape_and_slices=shape_and_slices, tensors=tensors)
