# RegexReplaceOp

import tensorflow as tf

replace_global = True
input = tf.constant("[abc,1]", shape=[2], dtype=tf.string)
pattern = tf.constant("[]", shape=[], dtype=tf.string)
rewrite = tf.constant("x", shape=[], dtype=tf.string)
tf.raw_ops.RegexReplace(input=input, pattern=pattern, rewrite=rewrite, replace_global=replace_global)
