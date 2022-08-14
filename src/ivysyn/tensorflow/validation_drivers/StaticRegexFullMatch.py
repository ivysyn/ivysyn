# StaticRegexFullMatchOp

import tensorflow as tf

pattern = "^s3://.*"
input = tf.constant("/tmp/convert_to_constants_testaf9vdmd2/tmpt1iczxut/frozen_saved_model/variables/variables", shape=[], dtype=tf.string)
tf.raw_ops.StaticRegexFullMatch(input=input, pattern=pattern)
