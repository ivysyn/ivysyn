# UnicodeDecodeBaseOp

import tensorflow as tf

input_encoding = "UTF-8"
errors = "replace"
replacement_char = 65533
replace_control_characters = False
Tsplits = tf.int64
input = tf.constant("仅今年前", shape=[1], dtype=tf.string)
tf.raw_ops.UnicodeDecode(input=input, input_encoding=input_encoding, errors=errors, replacement_char=replacement_char, replace_control_characters=replace_control_characters, Tsplits=Tsplits)
