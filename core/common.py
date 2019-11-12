#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : common.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 09:56:29
#   Description :
#
#================================================================

import tensorflow as tf
from core.relu_and_mask import leaky_relu_and_mask


def convolutional(input_data, filters_shape, trainable, name, prune_flag=tf.constant(False), downsample=False, activate=True, bn=True, data_format='NHWC'):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            if data_format == 'NCHW':
                paddings = tf.constant([[0, 0], [0, 0], [pad_h, pad_h], [pad_w, pad_w]])
                strides = (1, 1, 2, 2)
            else:
                paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
                strides = (1, 2, 2, 1)
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding, data_format=data_format)

        if bn:
            conv = tf.contrib.layers.batch_norm(conv, param_initializers={'beta': tf.zeros_initializer(),
                                                                              'gamma': tf.ones_initializer(),
                                                                              'moving_mean': tf.zeros_initializer(),
                                                                              'moving_variance': tf.ones_initializer()
                                                                              },
                                                                             is_training=trainable, data_format=data_format)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias,data_format=data_format)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)
        # nadav_wp_pruning-
        # if activate == True: conv = leaky_relu_and_mask(conv, alpha=0.1)

    return conv



def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name, data_format='NHWC'):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1', data_format=data_format)
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2', data_format=data_format)

        residual_output = input_data + short_cut

    return residual_output


def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv", data_format='NHWC'):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            if data_format == 'NCHW':
                input_data = tf.transpose(input_data, perm=[0, 2, 3, 1])
                input_shape = tf.shape(input_data)
                output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))
                output = tf.transpose(output, perm=[0, 3, 1, 2])
            else:
                input_shape = tf.shape(input_data)
                output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        if data_format == 'NCHW':
            output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',strides=(2,2),
                                                kernel_initializer=tf.random_normal_initializer(),
                                                data_format='channels_first')
        else:
            output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same', strides=(2, 2),
                                                kernel_initializer=tf.random_normal_initializer(),
                                                data_format='channels_last')

    return output