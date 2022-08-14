# BandedTriangularSolveOpCpu

import tensorflow as tf

lower = False
adjoint = False
matrix = tf.constant(1.41702199, shape=[2,2,2], dtype=tf.float32)
rhs = tf.constant(-0.206465051, shape=[2,2], dtype=tf.float32)
tf.raw_ops.BandedTriangularSolve(matrix=matrix, rhs=rhs, lower=lower, adjoint=adjoint)
