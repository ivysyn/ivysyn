# MfccOp

import tensorflow as tf

upper_frequency_limit = 4000
lower_frequency_limit = 20
filterbank_channel_count = 40
dct_coefficient_count = 13
spectrogram = tf.constant(1, shape=[1,1,513], dtype=tf.float32)
sample_rate = tf.constant(22050, shape=[], dtype=tf.int32)
tf.raw_ops.Mfcc(spectrogram=spectrogram, sample_rate=sample_rate, upper_frequency_limit=upper_frequency_limit, lower_frequency_limit=lower_frequency_limit, filterbank_channel_count=filterbank_channel_count, dct_coefficient_count=dct_coefficient_count)
