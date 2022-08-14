# BlockLSTMOp

import tensorflow as tf

cell_clip = 0
use_peephole = False
seq_len_max = tf.constant(1, shape=[], dtype=tf.int64)
x = tf.constant(0.504355371, shape=[1,1,1], dtype=tf.float32)
cs_prev = tf.constant(0.916770935, shape=[1,8], dtype=tf.float32)
h_prev = tf.constant(0.916770935, shape=[1,8], dtype=tf.float32)
w = tf.constant(0.184634328, shape=[9,32], dtype=tf.float32)
wci = tf.constant(0, shape=[8], dtype=tf.float32)
wcf = tf.constant(0, shape=[8], dtype=tf.float32)
wco = tf.constant(0, shape=[8], dtype=tf.float32)
b = tf.constant(0.173398286, shape=[32], dtype=tf.float32)
tf.raw_ops.BlockLSTMV2(seq_len_max=seq_len_max, x=x, cs_prev=cs_prev, h_prev=h_prev, w=w, wci=wci, wcf=wcf, wco=wco, b=b, cell_clip=cell_clip, use_peephole=use_peephole)
