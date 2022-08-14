# PrintV2Op

import tensorflow as tf

output_stream = "stderr"
end = "\n"
input = tf.constant("0.583637476", shape=[], dtype=tf.string)
tf.raw_ops.PrintV2(input=input, output_stream=output_stream, end=end)
