import numpy as np
import tensorflow as tf


def softmax_layer(inpt, weights, shape):
    fc_w = tf.constant(weights[0])
    fc_b = tf.constant(weights[1])

    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b,name = 'inner')

    return fc_h

def conv_layer(inpt, weights, filter_shape, stride):
    out_channels = filter_shape[3]

    filter_ = tf.constant(weights[0])
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.constant(weights[1])
    gamma = tf.constant(weights[2])
    
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    out = tf.nn.relu(batch_norm)

    return out

def residual_block(inpt, weights, output_depth, down_sample, projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer(inpt, weights[:3], [3, 3, input_depth, output_depth], 1)
    conv2 = conv_layer(conv1, weights[3:], [3, 3, output_depth, output_depth], 1)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer,weight3 = conv_layer(inpt, weights[6:9], [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer
    
    return res
