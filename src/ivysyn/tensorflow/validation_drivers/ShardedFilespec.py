# ShardedFilespecOp

import tensorflow as tf

basename = tf.constant("/tmp/saver_testnix_3upa/tmpidqhi0hk/sharded_basics", shape=[], dtype=tf.string)
num_shards = tf.constant(2, shape=[], dtype=tf.int32)
tf.raw_ops.ShardedFilespec(basename=basename, num_shards=num_shards)
