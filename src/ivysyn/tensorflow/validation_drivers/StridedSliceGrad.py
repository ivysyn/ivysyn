# StridedSliceGradOp

import tensorflow as tf

begin_mask = 0
end_mask = 0
ellipsis_mask = 0
new_axis_mask = 0
shrink_axis_mask = 1
shape = tf.constant([10,12,64], shape=[3], dtype=tf.int32)
begin = tf.constant(-1, shape=[1], dtype=tf.int32)
end = tf.constant(0, shape=[1], dtype=tf.int32)
strides = tf.constant(1, shape=[1], dtype=tf.int32)
dy = tf.constant(0, shape=[12,64], dtype=tf.float32)
tf.raw_ops.StridedSliceGrad(shape=shape, begin=begin, end=end, strides=strides, dy=dy, begin_mask=begin_mask, end_mask=end_mask, ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask)
