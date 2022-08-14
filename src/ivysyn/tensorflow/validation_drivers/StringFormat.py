# StringFormatOp

import tensorflow as tf

template = "{} [{}, \'x1\', {}]"
placeholder = "{}"
summarize = 10
inputs = tf.constant(0.583637476, shape=[3], dtype=tf.float32)
tf.raw_ops.StringFormat(inputs=inputs, template=template,
                        placeholder=placeholder, summarize=summarize)
