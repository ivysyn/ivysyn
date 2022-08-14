# StaticRegexReplaceOp

import tensorflow as tf

pattern = "\\d"
rewrite = "#"
replace_global = True
input = tf.constant("[hello,123,1+1]", shape=[3], dtype=tf.string)
tf.raw_ops.StaticRegexReplace(input=input, pattern=pattern, rewrite=rewrite, replace_global=replace_global)
