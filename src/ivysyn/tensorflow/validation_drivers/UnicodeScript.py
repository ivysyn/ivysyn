# UnicodeScriptOp

import tensorflow as tf

input = tf.constant([18,512,12412], shape=[3], dtype=tf.int32)
tf.raw_ops.UnicodeScript(input=input)
