# ReverseSequenceOp

import tensorflow as tf

seq_dim = 0
batch_dim = 2
input = tf.constant(1, shape=[4,2,3,1,1], dtype=tf.complex128)
seq_lengths = tf.constant([3,0,4], shape=[3], dtype=tf.int64)
tf.raw_ops.ReverseSequence(input=input, seq_lengths=seq_lengths, seq_dim=seq_dim, batch_dim=batch_dim)
