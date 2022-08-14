# DebugIdentityV2Op

import tensorflow as tf

tfdbg_context_id = "739c30e7-1716-4b83-a87f-b873dc3bbfcc"
op_name = "Const_2"
output_slot = 0
tensor_debug_mode = 5
debug_urls = ["file:///tmp/tmp5t5dn3l5"]
circular_buffer_size = 1000
tfdbg_run_id = "a3a8d861"
input = tf.constant(8, shape=[10], dtype=tf.float64)
tf.raw_ops.DebugIdentityV2(input=input, tfdbg_context_id=tfdbg_context_id, op_name=op_name, output_slot=output_slot, tensor_debug_mode=tensor_debug_mode, debug_urls=debug_urls, circular_buffer_size=circular_buffer_size, tfdbg_run_id=tfdbg_run_id)
