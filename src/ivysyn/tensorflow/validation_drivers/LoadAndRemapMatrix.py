# LoadAndRemapMatrixOp

import tensorflow as tf

num_rows = 7
num_cols = 16
max_rows_in_memory = -1
ckpt_path = tf.constant("/tmp/warm_starting_util_testpweab81v/tmpycdcktnw/model-0", shape=[], dtype=tf.string)
old_tensor_name = tf.constant("input_layer/sc_vocab_embedding/embedding_weights", shape=[], dtype=tf.string)
row_remapping = tf.constant([3,2,1], shape=[3], dtype=tf.int64)
col_remapping = tf.constant([], shape=[0], dtype=tf.int64)
initializing_values = tf.constant([], shape=[0,1], dtype=tf.float32)
tf.raw_ops.LoadAndRemapMatrix(ckpt_path=ckpt_path, old_tensor_name=old_tensor_name, row_remapping=row_remapping, col_remapping=col_remapping, initializing_values=initializing_values, num_rows=num_rows, num_cols=num_cols, max_rows_in_memory=max_rows_in_memory)
