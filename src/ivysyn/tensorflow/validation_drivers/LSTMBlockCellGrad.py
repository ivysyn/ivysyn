# LSTMBlockCellGradOp

import tensorflow as tf

use_peephole = False
x = tf.constant(0.837607, shape=[28,29], dtype=tf.float32)
cs_prev = tf.constant(0.592631638, shape=[28,17], dtype=tf.float32)
h_prev = tf.constant(0.592631638, shape=[28,17], dtype=tf.float32)
w = tf.constant(0.887386262, shape=[46,68], dtype=tf.float32)
wci = tf.constant(0, shape=[17], dtype=tf.float32)
wcf = tf.constant(0, shape=[17], dtype=tf.float32)
wco = tf.constant(0, shape=[17], dtype=tf.float32)
b = tf.constant(0.75259006, shape=[68], dtype=tf.float32)
i = tf.constant(1, shape=[28,17], dtype=tf.float32)
cs = tf.constant(1.59262991, shape=[28,17], dtype=tf.float32)
f = tf.constant(0.999997139, shape=[28,17], dtype=tf.float32)
o = tf.constant(0.999997735, shape=[28,17], dtype=tf.float32)
ci = tf.constant(1, shape=[28,17], dtype=tf.float32)
co = tf.constant(0.920551538, shape=[28,17], dtype=tf.float32)
cs_grad = tf.constant(0.756519377, shape=[28,17], dtype=tf.float32)
h_grad = tf.constant(0.579045, shape=[28,17], dtype=tf.float32)
tf.raw_ops.LSTMBlockCellGrad(x=x, cs_prev=cs_prev, h_prev=h_prev, w=w, wci=wci, wcf=wcf, wco=wco, b=b, i=i, cs=cs, f=f, o=o, ci=ci, co=co, cs_grad=cs_grad, h_grad=h_grad, use_peephole=use_peephole)
