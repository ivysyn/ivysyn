# StringLengthOp

import tensorflow as tf

unit = "BYTE"
input = tf.constant("[brown,fox,lazy,dog]", shape=[2], dtype=tf.string)
tf.raw_ops.StringLength(input=input, unit=unit)
