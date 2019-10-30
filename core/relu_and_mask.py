from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.contrib.framework.python.ops import variables

def relu_and_mask(x):
    mask = variables.model_variable('mask', shape=x.shape[-1:], dtype=dtypes.float32,
                                    initializer=init_ops.ones_initializer(),
                                    regularizer=None, collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
    ops.add_to_collections('masks', mask)
    x = x * mask
    return nn.relu(x)

def leaky_relu_and_mask(x, alpha=0.2):
    mask = variables.model_variable('mask', shape=x.shape[-1:], dtype=dtypes.float32,
                                    initializer=init_ops.ones_initializer(),
                                    regularizer=None,  collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
    ops.add_to_collections('masks', mask)
    x = x * mask
    return tf.nn.leaky_relu(x, alpha=alpha)
