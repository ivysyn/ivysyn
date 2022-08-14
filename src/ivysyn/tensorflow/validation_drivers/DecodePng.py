# DecodeImageV2Op

import tensorflow as tf

channels = 0
dtype = tf.uint8
contents = tf.constant("ThisIsNotAnImage!", shape=[], dtype=tf.string)
tf.raw_ops.DecodePng(contents=contents, channels=channels, dtype=dtype)
