import tensorflow as tf
from resnet import softmax_layer, conv_layer, residual_block

n_dict = {20:1, 32:2, 44:3, 56:4}
# ResNet architectures used for CIFAR-10
def resnet(inpt, n):
    if n < 20 or (n - 20) % 12 != 0:
        print "ResNet depth invalid."
        return

    num_conv = (n - 20) / 12 + 1
    layers = {}
    layer_list = []

    with tf.variable_scope('conv1'):
        conv1,conv1_w = conv_layer(inpt, [3, 3, 3, 16], 1)
        layers['conv1'] = conv1

        for weights in conv1_w:
          layer_list.append(weights)

    for i in range (num_conv):
        with tf.variable_scope('conv2_%d' % (i+1)):
            conv2_x, conv2_x_w = residual_block(layers['conv1'], 16, False)
            conv2,conv2_w = residual_block(conv2_x, 16, False)
            layers['conv2_x'] = conv2_x
            layers['conv2'] = conv2
            
            for weights in conv2_x_w:
              layer_list.append(weights)
            for weights in conv2_w:
              layer_list.append(weights)

    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i+1)):
            conv3_x,conv3_x_w = residual_block(layers['conv2'], 32, down_sample)
            conv3,conv3_w = residual_block(conv3_x, 32, False)
            layers['conv3_x'] = conv3_x
            layers['conv3'] = conv3
            
            for weights in conv3_x_w:
              layer_list.append(weights)
            for weights in conv3_w:
              layer_list.append(weights)

    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i+1)):
            conv4_x,conv4_x_w = residual_block(layers['conv3'], 64, down_sample)
            conv4,conv4_w = residual_block(conv4_x, 64, False)
            layers['conv4_x'] = conv4_x
            layers['conv4'] = conv4
            
            for weights in conv4_x_w:
              layer_list.append(weights)
            for weights in conv4_w:
              layer_list.append(weights)

    return layers, layer_list
