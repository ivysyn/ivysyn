# DecodeImageV2Op

import tensorflow as tf

channels = 0
contents = tf.constant("ThisIsNotAnImage!", shape=[], dtype=tf.string)
crop_window = tf.constant([1, 1, 1, 1], dtype=tf.int32)
tf.raw_ops.DecodeAndCropJpeg(contents=contents, crop_window=crop_window)
