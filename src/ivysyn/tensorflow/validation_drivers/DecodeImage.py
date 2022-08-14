# DecodeImageV2Op

import tensorflow as tf

channels = 0
dtype = tf.uint8
expand_animations = True
contents = tf.constant("ThisIsNotAnImage!", shape=[], dtype=tf.string)
tf.raw_ops.DecodeImage(contents=contents, channels=channels, dtype=dtype, expand_animations=expand_animations)
