# DebugNumericSummaryV2Op

import tensorflow as tf

output_dtype = tf.float64
tensor_debug_mode = 3
tensor_id = 247
input = tf.constant(1, shape=[], dtype=tf.float32)
tf.raw_ops.DebugNumericSummaryV2(input=input, output_dtype=output_dtype, tensor_debug_mode=tensor_debug_mode, tensor_id=tensor_id)
