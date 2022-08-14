# DecodeCSVOp

import tensorflow as tf

field_delim = ","
use_quote_delim = True
na_value = ""
select_cols = []
records = tf.constant("1,2,3", shape=[], dtype=tf.string)
record_defaults = tf.constant(1, shape=[1], dtype=tf.int32)
tf.raw_ops.DecodeCSV(records=records, record_defaults=record_defaults, field_delim=field_delim, use_quote_delim=use_quote_delim, na_value=na_value, select_cols=select_cols)
