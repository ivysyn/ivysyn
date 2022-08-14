# StatelessMultinomialOp

import tensorflow as tf

output_dtype = tf.int64
logits = tf.constant(0.583637476, shape=[3,4], dtype=tf.float32)
num_samples = tf.constant(3, shape=[], dtype=tf.int32)
seed = tf.constant([1,2], shape=[2], dtype=tf.int32)
tf.raw_ops.StatelessMultinomial(logits=logits, num_samples=num_samples, seed=seed, output_dtype=output_dtype)
