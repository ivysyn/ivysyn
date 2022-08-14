# BlockLSTMGradOp

import tensorflow as tf

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
i = tf.constant(0.947548032, shape=[1,1,8], dtype=tf.float32)
cs = tf.constant(1.82585835, shape=[1,1,8], dtype=tf.float32)
f = tf.constant(0.960281253, shape=[1,1,8], dtype=tf.float32)
o = tf.constant(0.952910185, shape=[1,1,8], dtype=tf.float32)
ci = tf.constant(0.997839093, shape=[1,1,8], dtype=tf.float32)
co = tf.constant(0.949419379, shape=[1,1,8], dtype=tf.float32)
h = tf.constant(0.904711425, shape=[1,1,8], dtype=tf.float32)
cs_grad = tf.constant(1, shape=[1,1,8], dtype=tf.float32)
h_grad = tf.constant(1, shape=[1,1,8], dtype=tf.float32)
tf.raw_ops.BlockLSTMGradV2(seq_len_max=seq_len_max, x=x, cs_prev=cs_prev, h_prev=h_prev, w=w, wci=wci, wcf=wcf, wco=wco, b=b, i=i, cs=cs, f=f, o=o, ci=ci, co=co, h=h, cs_grad=cs_grad, h_grad=h_grad, use_peephole=use_peephole)
