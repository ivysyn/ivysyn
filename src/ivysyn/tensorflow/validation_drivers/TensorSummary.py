# SummaryTensorOp

import tensorflow as tf

description = "tensor_summary"
labels = ["3", "summary"]
display_name = "test"
tensor = tf.constant(3, shape=[], dtype=tf.float32)
tf.raw_ops.TensorSummary(tensor=tensor, description=description, labels=labels, display_name=display_name)
