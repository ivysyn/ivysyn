# ShardedFilenameOp

import tensorflow as tf

basename = tf.constant("/tmp/convert_to_constants_testaf9vdmd2/tmpt1iczxut/frozen_saved_model/variables/variables_temp/part", shape=[], dtype=tf.string)
shard = tf.constant(0, shape=[], dtype=tf.int32)
num_shards = tf.constant(1, shape=[], dtype=tf.int32)
tf.raw_ops.ShardedFilename(basename=basename, shard=shard, num_shards=num_shards)
