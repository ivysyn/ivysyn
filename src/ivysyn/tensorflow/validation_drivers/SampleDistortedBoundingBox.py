# StatefulSampleDistortedBoundingBoxOp

import tensorflow as tf

seed = 0
seed2 = 0
aspect_ratio_range = [0.75, 1.33]
area_range = [0.05, 1]
max_attempts = 100
use_image_if_no_bounding_boxes = False
image_size = tf.constant([40, 50, 1], shape=[3], dtype=tf.int32)
bounding_boxes = tf.constant(0, shape=[1, 1, 4], dtype=tf.float32)
min_object_covered = 0.1
tf.raw_ops.SampleDistortedBoundingBox(image_size=image_size, bounding_boxes=bounding_boxes, seed=seed, seed2=seed2, min_object_covered=min_object_covered,
                                      aspect_ratio_range=aspect_ratio_range, area_range=area_range, max_attempts=max_attempts, use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes)
