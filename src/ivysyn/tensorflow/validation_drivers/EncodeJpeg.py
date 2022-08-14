# EncodeJpegOp

import tensorflow as tf

format = ""
quality = 95
progressive = False
optimize_size = False
chroma_downsampling = True
density_unit = "in"
x_density = 300
y_density = 300
xmp_metadata = ""
image = tf.constant(0, shape=[200,256,3], dtype=tf.uint8)
tf.raw_ops.EncodeJpeg(image=image, format=format, quality=quality, progressive=progressive, optimize_size=optimize_size, chroma_downsampling=chroma_downsampling, density_unit=density_unit, x_density=x_density, y_density=y_density, xmp_metadata=xmp_metadata)
