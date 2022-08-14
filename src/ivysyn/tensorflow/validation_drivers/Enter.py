# EnterOp

import tensorflow as tf

frame_name = "StatefulPartitionedCall/rnn/while"
is_constant = False
parallel_iterations = 32
data = tf.constant(10, shape=[], dtype=tf.int32)
tf.raw_ops.Enter(data=data, frame_name=frame_name,
                 is_constant=is_constant, parallel_iterations=parallel_iterations)
