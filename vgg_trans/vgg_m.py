# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np

def const_vgg_net(input_image, layer_list):
    net = {}
    bgr = input_image
    
    #bgr = tf.contrib.layers.batch_norm(bgr)
    conv1_1 = const_conv_layer(bgr, layer_list[0:3],)
    
    relu1_1 = tf.nn.relu(conv1_1) 
    conv1_2 = const_conv_layer(relu1_1, layer_list[3:6])
    
    relu1_2 = tf.nn.relu(conv1_2) 
    pool1 = max_pool(relu1_2, 'pool1')

    conv2_1 = const_conv_layer(pool1, layer_list[6:9])
    
    relu2_1 = tf.nn.relu(conv2_1) 
    conv2_2 = const_conv_layer(relu2_1, layer_list[9:12])
    
    relu2_2 = tf.nn.relu(conv2_2) 
    pool2 = max_pool(relu2_2, 'pool2')

    conv3_1 = const_conv_layer(pool2, layer_list[12:15])
    
    relu3_1 = tf.nn.relu(conv3_1) 
    conv3_2 = const_conv_layer(relu3_1, layer_list[15:18])
    
    relu3_2 = tf.nn.relu(conv3_2) 
    conv3_3 = const_conv_layer(relu3_2,layer_list[18:21])
    
    relu3_3 = tf.nn.relu(conv3_3) 
    conv3_4 = const_conv_layer(relu3_3,layer_list[21:24])
    
    relu3_4 = tf.nn.relu(conv3_4) 
    pool3 = max_pool(relu3_4, 'pool3')

    conv4_1 = const_conv_layer(pool3,layer_list[24:27])
    
    relu4_1 = tf.nn.relu(conv4_1) 
    conv4_2 = const_conv_layer(relu4_1,layer_list[27:30])
    
    relu4_2 = tf.nn.relu(conv4_2) 
    conv4_3 = const_conv_layer(relu4_2,layer_list[30:33])
    
    relu4_3 = tf.nn.relu(conv4_3) 
    conv4_4 = const_conv_layer(relu4_3,layer_list[33:36])
    
    relu4_4 = tf.nn.relu(conv4_4) 
    pool4 = max_pool(relu4_4, 'pool4')

    conv5_1 = const_conv_layer(pool4,layer_list[36:39])
    
    relu5_1 = tf.nn.relu(conv5_1) 
    conv5_2 = const_conv_layer(relu5_1,layer_list[39:42])
    
    relu5_2 = tf.nn.relu(conv5_2) 
    conv5_3 = const_conv_layer(relu5_2,layer_list[42:45])
    
    relu5_3 = tf.nn.relu(conv5_3) 
    conv5_4 = const_conv_layer(relu5_3,layer_list[45:48])
    
    relu5_4 = tf.nn.relu(conv5_4) 
    pool5 = max_pool(relu5_4, 'pool5')
 
    net['conv1_1'] = conv1_1
    net['conv1_2'] = conv1_2
    net['conv2_1'] = conv2_1
    net['conv2_2'] = conv2_2
    net['conv3_1'] = conv3_1
    net['conv3_2'] = conv3_2
    net['conv3_3'] = conv3_3
    net['conv3_4'] = conv3_4
    net['conv4_1'] = conv4_1
    net['conv4_2'] = conv4_2
    net['conv4_3'] = conv4_3
    net['conv4_4'] = conv4_4
    net['conv5_1'] = conv5_1
    net['conv5_2'] = conv5_2
    net['conv5_3'] = conv5_3
    net['conv5_4'] = conv5_4
    net['relu1_1'] = relu1_1
    net['relu1_2'] = relu1_2
    net['relu2_1'] = relu2_1
    net['relu2_2'] = relu2_2
    net['relu3_1'] = relu3_1
    net['relu3_2'] = relu3_2
    net['relu3_3'] = relu3_3
    net['relu3_4'] = relu3_4
    net['relu4_1'] = relu4_1
    net['relu4_2'] = relu4_2
    net['relu4_3'] = relu4_3
    net['relu4_4'] = relu4_4
    net['relu5_1'] = relu5_1
    net['relu5_2'] = relu5_2
    net['relu5_3'] = relu5_3
    net['relu5_4'] = relu5_4
    net['relu5_4'] = relu5_4
    net['pool1'] = pool1 
    net['pool2'] = pool2
    net['pool3'] = pool3
    net['pool4'] = pool4
    net['pool5'] = pool5
    
    return net

def vgg_net(input_image):
    net = {}
    layer_list = []
    bgr = input_image
    
    #bgr = tf.contrib.layers.batch_norm(bgr)
    conv1_1,w1 = conv_layer(bgr, 3, 64, "conv1_1")
    
    relu1_1 = tf.nn.relu(conv1_1) 
    conv1_2,w2 = conv_layer(relu1_1, 64, 64, "conv1_2")
    
    relu1_2 = tf.nn.relu(conv1_2) 
    pool1 = max_pool(relu1_2, 'pool1')

    conv2_1, w3 = conv_layer(pool1, 64, 128, "conv2_1")
    
    relu2_1 = tf.nn.relu(conv2_1) 
    conv2_2, w4 = conv_layer(relu2_1, 128, 128, "conv2_2")
    
    relu2_2 = tf.nn.relu(conv2_2) 
    pool2 = max_pool(relu2_2, 'pool2')

    conv3_1, w5 = conv_layer(pool2, 128, 256, "conv3_1")
    
    relu3_1 = tf.nn.relu(conv3_1) 
    conv3_2, w6 = conv_layer(relu3_1, 256, 256, "conv3_2")
    
    relu3_2 = tf.nn.relu(conv3_2) 
    conv3_3, w7 = conv_layer(relu3_2, 256, 256, "conv3_3")
    
    relu3_3 = tf.nn.relu(conv3_3) 
    conv3_4, w8 = conv_layer(relu3_3, 256, 256, "conv3_4")
    
    relu3_4 = tf.nn.relu(conv3_4) 
    pool3 = max_pool(relu3_4, 'pool3')

    conv4_1, w9 = conv_layer(pool3, 256, 512, "conv4_1")
    
    relu4_1 = tf.nn.relu(conv4_1) 
    conv4_2, w10 = conv_layer(relu4_1, 512, 512, "conv4_2")
    
    relu4_2 = tf.nn.relu(conv4_2) 
    conv4_3, w11 = conv_layer(relu4_2, 512, 512, "conv4_3")
    
    relu4_3 = tf.nn.relu(conv4_3) 
    conv4_4, w12 = conv_layer(relu4_3, 512, 512, "conv4_4")

    relu4_4 = tf.nn.relu(conv4_4) 
    pool4 = max_pool(relu4_4, 'pool4')

    conv5_1, w13 = conv_layer(pool4, 512, 512, "conv5_1")

    relu5_1 = tf.nn.relu(conv5_1) 
    conv5_2, w14 = conv_layer(relu5_1, 512, 512, "conv5_2")

    relu5_2 = tf.nn.relu(conv5_2) 
    conv5_3, w15 = conv_layer(relu5_2, 512, 512, "conv5_3")

    relu5_3 = tf.nn.relu(conv5_3) 
    conv5_4, w16 = conv_layer(relu5_3, 512, 512, "conv5_4")

    relu5_4 = tf.nn.relu(conv5_4) 
    pool5 = max_pool(relu5_4, 'pool5')
    
    net['conv1_1'] = conv1_1
    net['conv1_2'] = conv1_2
    net['conv2_1'] = conv2_1
    net['conv2_2'] = conv2_2
    net['conv3_1'] = conv3_1
    net['conv3_2'] = conv3_2
    net['conv3_3'] = conv3_3
    net['conv3_4'] = conv3_4
    net['conv4_1'] = conv4_1
    net['conv4_2'] = conv4_2
    net['conv4_3'] = conv4_3
    net['conv4_4'] = conv4_4
    net['conv5_1'] = conv5_1
    net['conv5_2'] = conv5_2
    net['conv5_3'] = conv5_3
    net['conv5_4'] = conv5_4
    net['relu1_1'] = relu1_1
    net['relu1_2'] = relu1_2
    net['relu2_1'] = relu2_1
    net['relu2_2'] = relu2_2
    net['relu3_1'] = relu3_1
    net['relu3_2'] = relu3_2
    net['relu3_3'] = relu3_3
    net['relu3_4'] = relu3_4
    net['relu4_1'] = relu4_1
    net['relu4_2'] = relu4_2
    net['relu4_3'] = relu4_3
    net['relu4_4'] = relu4_4
    net['relu5_1'] = relu5_1
    net['relu5_2'] = relu5_2
    net['relu5_3'] = relu5_3
    net['relu5_4'] = relu5_4
    net['relu5_4'] = relu5_4
    net['pool1'] = pool1 
    net['pool2'] = pool2
    net['pool3'] = pool3
    net['pool4'] = pool4
    net['pool5'] = pool5
    layer_list.append(w1[0])
    layer_list.append(w1[1])
    layer_list.append(w1[2])

    layer_list.append(w2[0])
    layer_list.append(w2[1])
    layer_list.append(w2[2])
    
    layer_list.append(w3[0])
    layer_list.append(w3[1])
    layer_list.append(w3[2])
    
    layer_list.append(w4[0])
    layer_list.append(w4[1])
    layer_list.append(w4[2])
    
    layer_list.append(w5[0])
    layer_list.append(w5[1])
    layer_list.append(w5[2])
    
    layer_list.append(w6[0])
    layer_list.append(w6[1])
    layer_list.append(w6[2])
    
    layer_list.append(w7[0])
    layer_list.append(w7[1])
    layer_list.append(w7[2])
    
    layer_list.append(w8[0])
    layer_list.append(w8[1])
    layer_list.append(w8[2])
    
    layer_list.append(w9[0])
    layer_list.append(w9[1])
    layer_list.append(w9[2])
    
    layer_list.append(w10[0])
    layer_list.append(w10[1])
    layer_list.append(w10[2])
    
    layer_list.append(w11[0])
    layer_list.append(w11[1])
    layer_list.append(w11[2])
    
    layer_list.append(w12[0])
    layer_list.append(w12[1])
    layer_list.append(w12[2])
    
    layer_list.append(w13[0])
    layer_list.append(w13[1])
    layer_list.append(w13[2])
    
    layer_list.append(w14[0])
    layer_list.append(w14[1])
    layer_list.append(w14[2])
    
    layer_list.append(w15[0])
    layer_list.append(w15[1])
    layer_list.append(w15[2])
    
    layer_list.append(w16[0])
    layer_list.append(w16[1])
    layer_list.append(w16[2])
    
    return net, layer_list


def vgg_train_net(input_image, classes):
    net = {}
    layer_list = []
    bgr = input_image

    #bgr = tf.contrib.layers.batch_norm(bgr)
    conv1_1,w1 = conv_layer(bgr, 3, 64, "conv1_1")
    
    relu1_1 = tf.nn.relu(conv1_1) 
    conv1_2,w2 = conv_layer(relu1_1, 64, 64, "conv1_2")
    
    relu1_2 = tf.nn.relu(conv1_2) 
    pool1 = max_pool(relu1_2, 'pool1')

    conv2_1, w3 = conv_layer(pool1, 64, 128, "conv2_1")
    
    relu2_1 = tf.nn.relu(conv2_1) 
    conv2_2, w4 = conv_layer(relu2_1, 128, 128, "conv2_2")
    
    relu2_2 = tf.nn.relu(conv2_2) 
    pool2 = max_pool(relu2_2, 'pool2')

    conv3_1, w5 = conv_layer(pool2, 128, 256, "conv3_1")
    
    relu3_1 = tf.nn.relu(conv3_1) 
    conv3_2, w6 = conv_layer(relu3_1, 256, 256, "conv3_2")
    
    relu3_2 = tf.nn.relu(conv3_2) 
    conv3_3, w7 = conv_layer(relu3_2, 256, 256, "conv3_3")
    
    relu3_3 = tf.nn.relu(conv3_3) 
    conv3_4, w8 = conv_layer(relu3_3, 256, 256, "conv3_4")
    
    relu3_4 = tf.nn.relu(conv3_4) 
    pool3 = max_pool(relu3_4, 'pool3')

    conv4_1, w9 = conv_layer(pool3, 256, 512, "conv4_1")
    
    relu4_1 = tf.nn.relu(conv4_1) 
    conv4_2, w10 = conv_layer(relu4_1, 512, 512, "conv4_2")
    
    relu4_2 = tf.nn.relu(conv4_2) 
    conv4_3, w11 = conv_layer(relu4_2, 512, 512, "conv4_3")
    
    relu4_3 = tf.nn.relu(conv4_3) 
    conv4_4, w12 = conv_layer(relu4_3, 512, 512, "conv4_4")

    relu4_4 = tf.nn.relu(conv4_4) 
    pool4 = max_pool(relu4_4, 'pool4')

    conv5_1, w13 = conv_layer(pool4, 512, 512, "conv5_1")

    relu5_1 = tf.nn.relu(conv5_1) 
    conv5_2, w14 = conv_layer(relu5_1, 512, 512, "conv5_2")

    relu5_2 = tf.nn.relu(conv5_2) 
    conv5_3, w15 = conv_layer(relu5_2, 512, 512, "conv5_3")
    
    relu5_3 = tf.nn.relu(conv5_3) 
    conv5_4, w16 = conv_layer(relu5_3, 512, 512, "conv5_4")

    relu5_4 = tf.nn.relu(conv5_4) 
    pool5 = max_pool(relu5_4, 'pool5')
    
    saver = tf.train.Saver()
    in_size = ((int(bgr.get_shape()[1]) // (2 ** 5)) ** 2) * 512

    fc6, w17 = fc_layer(pool5, in_size, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
    relu6 = tf.nn.relu(fc6) 

    fc7, w18 = fc_layer(relu6, 4096, 4096, "fc7")
    relu7 = tf.nn.relu(fc7)

    fc8, w19 = fc_layer(relu7, 4096, classes, "fc8")

    prob = tf.nn.softmax(fc8, name="prob")

    return fc8, prob, saver


def const_conv_layer(input, weights):
    conv = tf.nn.conv2d(input, tf.constant(weights[0]), strides=(1, 1, 1, 1),
            padding='SAME')
    bias = tf.constant(weights[1])

    conv = tf.nn.bias_add(conv, bias)

    mean, var = tf.nn.moments(conv, axes=[0,1,2])                                                                                                          
    beta = tf.constant(weights[2])   

    gamma = None    

    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    return batch_norm

def const_fc_layer(input, weights, in_size, out_size, name):
    name_w = name + '_w'
    weight = tf.constant(weights[0], name=name_w)
    
    name_b = name + '_b'
    bias = tf.constant(weights[1], name=name_b)
    
    x = tf.reshape(input, [-1, in_size])
    fc = tf.nn.bias_add(tf.matmul(x, weight), bias)
    
    return fc

def conv_layer(input, in_channel, out_channel, name):
    initial_value = tf.truncated_normal([3, 3, in_channel, out_channel], stddev=0.01)
    name_w = name + '_w'
    weight = tf.Variable(initial_value, name=name_w)
    conv = tf.nn.conv2d(input, weight, strides=(1, 1, 1, 1), padding='SAME')

    initial_value = tf.truncated_normal([out_channel], stddev=0.01)
    name_b = name + '_b'
    bias = tf.Variable(initial_value, name=name_b)

    conv = tf.nn.bias_add(conv, bias)

    mean, var = tf.nn.moments(conv, axes=[0,1,2])                                                                                                          
    beta = tf.Variable(tf.zeros([out_channel]), name="beta")         

    initial = tf.truncated_normal([out_channel], stddev=0.01)
    #gamma = tf.Variable(initial, name="gamma")         
    gamma = None      

    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    return batch_norm, (weight, bias, beta) 


def fc_layer(input, in_size, out_size, name):
    initial_value = tf.truncated_normal([in_size, out_size], stddev=0.001)
    name_w = name + '_w'
    weight = tf.Variable(initial_value, name=name_w)
    
    initial_value = tf.truncated_normal([out_size], stddev=0.001)
    name_b = name + '_b'
    bias = tf.Variable(initial_value, name=name_b)
    
    x = tf.reshape(input, [-1, in_size])
    fc = tf.nn.bias_add(tf.matmul(x, weight), bias)
    
    return fc, (weight, bias)

def avg_pool(input, name):
        return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool(input, name):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)   
