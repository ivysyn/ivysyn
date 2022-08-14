# LSTMBlockCellOp

import tensorflow as tf

forget_bias = 1
cell_clip = 0
use_peephole = False
x = tf.constant(0.837607, shape=[28,29], dtype=tf.float32)
cs_prev = tf.constant(0.592631638, shape=[28,17], dtype=tf.float32)
h_prev = tf.constant(0.592631638, shape=[28,17], dtype=tf.float32)
w = tf.constant(0.887386262, shape=[46,68], dtype=tf.float32)
wci = tf.constant(0, shape=[17], dtype=tf.float32)
wcf = tf.constant(0, shape=[17], dtype=tf.float32)
wco = tf.constant(0, shape=[17], dtype=tf.float32)
b = tf.constant(0.75259006, shape=[68], dtype=tf.float32)
tf.raw_ops.LSTMBlockCell(x=x, cs_prev=cs_prev, h_prev=h_prev, w=w, wci=wci, wcf=wcf, wco=wco, b=b, forget_bias=forget_bias, cell_clip=cell_clip, use_peephole=use_peephole)
