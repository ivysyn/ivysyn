# DeleteSessionTensorOp

import tensorflow as tf

handle = tf.constant("GetSessionHandle;0;/job", shape=[], dtype=tf.string)
tf.raw_ops.DeleteSessionTensor(handle=handle)
