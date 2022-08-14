# StatefulMultinomialOp

import tensorflow as tf

seed = 87654321
seed2 = 0
output_dtype = tf.int64
logits = tf.constant(1, shape=[1,2], dtype=tf.float32)
num_samples = tf.constant(15, shape=[], dtype=tf.int32)
tf.raw_ops.Multinomial(logits=logits, num_samples=num_samples, seed=seed, seed2=seed2, output_dtype=output_dtype)
