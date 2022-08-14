# GetSessionTensorOp

import tensorflow as tf

dtype = tf.float32
handle = tf.constant("GetSessionHandle;0;/job", shape=[], dtype=tf.string)
tf.raw_ops.GetSessionTensor(handle=handle, dtype=dtype)
