# MergeV2Checkpoints

import tensorflow as tf

delete_old_dirs = True
checkpoint_prefixes = tf.constant("-00000", shape=[1], dtype=tf.string)
destination_prefix = tf.constant("/tmp/convert_to_constants_testaf9vdmd2/tmpt1iczxut/frozen_saved_model/variables/variables", shape=[], dtype=tf.string)
tf.raw_ops.MergeV2Checkpoints(checkpoint_prefixes=checkpoint_prefixes, destination_prefix=destination_prefix, delete_old_dirs=delete_old_dirs)
