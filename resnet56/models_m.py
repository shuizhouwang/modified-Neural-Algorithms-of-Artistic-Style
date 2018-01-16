import tensorflow as tf
from resnet_m import softmax_layer, conv_layer, residual_block

n_dict = {20:1, 32:2, 44:3, 56:4}
# ResNet architectures used for CIFAR-10
def resnet_m(inpt, n, copy):
    if n < 20 or (n - 20) % 12 != 0:
        print "ResNet depth invalid."
        return

    num_conv = (n - 20) / 12 + 1
    layers = {}
    layer_list = []
    cursor = 0

    with tf.variable_scope('conv1'):
        
        weights = []
        for j in range(3):
          weights.append(copy[cursor])
          cursor += 1
        conv1 = conv_layer(inpt, weights , [3, 3, 3, 16], 1)
        
        layer_list.append(conv1)
        layers['conv1'] = conv1

    for i in range (num_conv):
        with tf.variable_scope('conv2_%d' % (i+1)):
            weights = []
            for j in range(6):
              weights.append(copy[cursor])
              cursor += 1
            conv2_x = residual_block(layer_list[-1],weights, 16, False)
            
            weights = []
            for j in range(6):
              weights.append(copy[cursor])
              cursor += 1
            conv2 = residual_block(conv2_x, weights, 16, False)
            
            layer_list.append(conv2_x)
            layer_list.append(conv2)
            layers['conv2_x_' + str(i)] = conv2_x
            layers['conv2_' + str(i)] = conv2
            
    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i+1)):
            weights = []
            for j in range(6):
              weights.append(copy[cursor])
              cursor += 1
            conv3_x = residual_block(layer_list[-1], weights, 32, down_sample)
            
            weights = []
            for j in range(6):
              weights.append(copy[cursor])
              cursor += 1
            conv3 = residual_block(conv3_x, weights, 32, False)
            
            layer_list.append(conv3_x)
            layer_list.append(conv3)
            layers['conv3_x_' + str(i)] = conv3_x
            layers['conv3_' + str(i)] = conv3
            
    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i+1)): 
            weights = []
            for j in range(6):
              weights.append(copy[cursor])
              cursor += 1
            conv4_x = residual_block(layer_list[-1], weights, 64, down_sample)
            
            weights = []
            for j in range(6):
              weights.append(copy[cursor])
              cursor += 1
            conv4 = residual_block(conv4_x, weights, 64, False)
            
            layer_list.append(conv4_x)
            layer_list.append(conv4)
            layers['conv4_x_' + str(i)] = conv4_x
            layers['conv4_' + str(i)] = conv4

    return layers
