# SparseCrossOp

import tensorflow as tf

hashed_output = True
num_buckets = 1000
hash_key = 956888297470
out_type = tf.int64
internal_type = tf.string
indices = [tf.constant(0, shape=[3, 2], dtype=tf.int64), tf.constant(
    0, shape=[1, 2], dtype=tf.int64), tf.constant(0, shape=[2, 2], dtype=tf.int64)]
values = [tf.constant("[batch1-FC1-F1,batch1-FC1-F2,batch1-FC1-F3]", shape=[3], dtype=tf.string), tf.constant("[batch1-FC1-F1,batch1-FC1-F2,batch1-FC1-F3]",
                                                                                                              shape=[1], dtype=tf.string), tf.constant("[batch1-FC1-F1,batch1-FC1-F2,batch1-FC1-F3]", shape=[2], dtype=tf.string)]
shapes = [tf.constant(1, shape=[2], dtype=tf.int64), tf.constant(
    1, shape=[2], dtype=tf.int64), tf.constant(1, shape=[2], dtype=tf.int64)]
dense_inputs = []
tf.raw_ops.SparseCross(indices=indices, values=values, shapes=shapes, dense_inputs=dense_inputs, hashed_output=hashed_output,
                       num_buckets=num_buckets, hash_key=hash_key, out_type=out_type, internal_type=internal_type)
