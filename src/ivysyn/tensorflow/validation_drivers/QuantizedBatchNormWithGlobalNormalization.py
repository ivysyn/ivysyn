# QuantizedBatchNormOp

import tensorflow as tf

out_type = tf.qint32
variance_epsilon = 0.001
scale_after_normalization = False
t = tf.constant(129, shape=[1,1,6,2], dtype=tf.quint8)
t_min = tf.constant(-128, shape=[1], dtype=tf.float32)
t_max = tf.constant(127, shape=[1], dtype=tf.float32)
m = tf.constant([128,255], shape=[2], dtype=tf.quint8)
m_min = tf.constant(0, shape=[1], dtype=tf.float32)
m_max = tf.constant(20, shape=[1], dtype=tf.float32)
v = tf.constant([64,128], shape=[2], dtype=tf.quint8)
v_min = tf.constant(0, shape=[1], dtype=tf.float32)
v_max = tf.constant(1, shape=[1], dtype=tf.float32)
beta = tf.constant([26,153], shape=[2], dtype=tf.quint8)
beta_min = tf.constant(0, shape=[1], dtype=tf.float32)
beta_max = tf.constant(1, shape=[1], dtype=tf.float32)
gamma = tf.constant([0,0], shape=[2], dtype=tf.quint8)
gamma_min = tf.constant(0, shape=[1], dtype=tf.float32)
gamma_max = tf.constant(1, shape=[1], dtype=tf.float32)
tf.raw_ops.QuantizedBatchNormWithGlobalNormalization(t=t, t_min=t_min, t_max=t_max, m=m, m_min=m_min, m_max=m_max, v=v, v_min=v_min, v_max=v_max, beta=beta, beta_min=beta_min, beta_max=beta_max, gamma=gamma, gamma_min=gamma_min, gamma_max=gamma_max, out_type=out_type, variance_epsilon=variance_epsilon, scale_after_normalization=scale_after_normalization)
