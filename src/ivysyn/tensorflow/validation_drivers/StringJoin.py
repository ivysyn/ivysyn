# StringJoinOp

import tensorflow as tf

separator = ""
inputs = [tf.constant("/tmp/convert_to_constants_testaf9vdmd2/tmpt1iczxut/frozen_saved_model/variables/variables",
                      shape=[], dtype=tf.string), tf.constant("_temp/part", shape=[], dtype=tf.string)]
tf.raw_ops.StringJoin(inputs=inputs, separator=separator)
