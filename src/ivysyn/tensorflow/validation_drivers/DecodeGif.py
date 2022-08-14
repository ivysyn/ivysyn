# DecodeImageV2Op

import tensorflow as tf

contents = tf.constant("ThisIsNotAnImage!", shape=[], dtype=tf.string)
tf.raw_ops.DecodeGif(contents=contents)
