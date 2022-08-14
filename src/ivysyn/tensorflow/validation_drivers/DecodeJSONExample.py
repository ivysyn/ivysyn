# DecodeJSONExampleOp

import tensorflow as tf

json_examples = tf.constant("{\n", shape=[2,1], dtype=tf.string)
tf.raw_ops.DecodeJSONExample(json_examples=json_examples)
