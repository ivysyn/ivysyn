# CTCBeamSearchDecoderOp

import tensorflow as tf

beam_width = 2
top_paths = 2
merge_repeated = True
inputs = tf.constant(2.30999, shape=[7,1,6], dtype=tf.float32)
sequence_length = tf.constant(5, shape=[1], dtype=tf.int32)
tf.raw_ops.CTCBeamSearchDecoder(inputs=inputs, sequence_length=sequence_length, beam_width=beam_width, top_paths=top_paths, merge_repeated=merge_repeated)
