# RestoreSliceOp

import tensorflow as tf

dt = tf.int32
preferred_shard = -1
file_pattern = tf.constant("/home/ivyusr/.cache/bazel/_bazel_ivyusr/66cba586b2867519e0c64ff63029eaa9/execroot/org_tensorflow/_tmp/aa8a8c78961330944213844b04568343/tensor_int", shape=[], dtype=tf.string)
tensor_name = tf.constant("tensor_int", shape=[], dtype=tf.string)
shape_and_slice = tf.constant("4", shape=[], dtype=tf.string)
tf.raw_ops.RestoreSlice(file_pattern=file_pattern, tensor_name=tensor_name, shape_and_slice=shape_and_slice, dt=dt, preferred_shard=preferred_shard)
