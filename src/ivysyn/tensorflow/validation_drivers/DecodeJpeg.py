# DecodeImageV2Op

import tensorflow as tf

channels = 0
contents = tf.constant("ThisIsNotAnImage!", shape=[], dtype=tf.string)
tf.raw_ops.DecodeJpeg(contents=contents, channels=channels)
