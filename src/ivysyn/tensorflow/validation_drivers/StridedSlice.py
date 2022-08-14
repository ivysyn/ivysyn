# StridedSliceOp

import tensorflow as tf

begin_mask = 0
end_mask = 0
ellipsis_mask = 0
new_axis_mask = 0
shrink_axis_mask = 1
input = tf.constant([5,2], shape=[2], dtype=tf.int32)
begin = tf.constant(0, shape=[1], dtype=tf.int32)
end = tf.constant(1, shape=[1], dtype=tf.int32)
strides = tf.constant(1, shape=[1], dtype=tf.int32)
tf.raw_ops.StridedSlice(input=input, begin=begin, end=end, strides=strides, begin_mask=begin_mask, end_mask=end_mask, ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask)
