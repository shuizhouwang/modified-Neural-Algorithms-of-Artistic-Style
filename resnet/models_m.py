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
    cursor = 0

    with tf.variable_scope('conv1'):
        
        weights = []
        for j in range(3):
          weights.append(copy[cursor])
          cursor += 1
        conv1 = conv_layer(inpt, weights , [3, 3, 3, 16], 1)
        
        layers['conv1'] = conv1

    for i in range (num_conv):
        with tf.variable_scope('conv2_%d' % (i+1)):
            weights = []
            for j in range(6):
              weights.append(copy[cursor])
              cursor += 1
            conv2_x = residual_block(layers['conv1'],weights, 16, False)
            
            weights = []
            for j in range(6):
              weights.append(copy[cursor])
              cursor += 1
            conv2 = residual_block(conv2_x, weights, 16, False)
            
            layers['conv2_x'] = conv2_x
            layers['conv2'] = conv2
            
    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i+1)):
            weights = []
            for j in range(6):
              weights.append(copy[cursor])
              cursor += 1
            conv3_x = residual_block(layers['conv2'], weights, 32, down_sample)
            
            weights = []
            for j in range(6):
              weights.append(copy[cursor])
              cursor += 1
            conv3 = residual_block(conv3_x, weights, 32, False)
            
            layers['conv3_x'] = conv3_x
            layers['conv3'] = conv3
            
    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i+1)): 
            weights = []
            for j in range(6):
              weights.append(copy[cursor])
              cursor += 1
            conv4_x = residual_block(layers['conv3'], weights, 64, down_sample)
            
            weights = []
            for j in range(6):
              weights.append(copy[cursor])
              cursor += 1
            conv4 = residual_block(conv4_x, weights, 64, False)
            
            layers['conv4_x'] = conv4_x
            layers['conv4'] = conv4

    return layers
