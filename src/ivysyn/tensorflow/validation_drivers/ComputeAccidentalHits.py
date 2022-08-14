# ComputeAccidentalHitsOp

import tensorflow as tf

num_true = 1
seed = 87654321
seed2 = 1503137960
true_classes = tf.constant(403, shape=[3,1], dtype=tf.int64)
sampled_candidates = tf.constant(1, shape=[4], dtype=tf.int64)
tf.raw_ops.ComputeAccidentalHits(true_classes=true_classes, sampled_candidates=sampled_candidates, num_true=num_true, seed=seed, seed2=seed2)
