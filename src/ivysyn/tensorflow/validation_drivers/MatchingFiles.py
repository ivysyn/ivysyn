# MatchingFilesOp

import tensorflow as tf

pattern = tf.constant(".*", shape=[], dtype=tf.string)
tf.raw_ops.MatchingFiles(pattern=pattern)
