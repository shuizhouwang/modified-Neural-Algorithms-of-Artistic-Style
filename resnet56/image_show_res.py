import tensorflow as tf
import scipy.misc
import numpy as np

from models import resnet

def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img

input_image = np.expand_dims(imread('./dog.jpg'), axis=0).astype('float32')

net,_ = resnet(input_image, 56)
saver = tf.train.Saver()

for keys in net:
  if keys == 'output':
    continue
  output = net[keys]
  output_shape = int(output.get_shape()[3])
  output_split = tf.split(output, num_or_size_splits=output_shape, axis=3)

  feature_list = []
  for i in range(output_shape):
    feature_list.append(output_split[i])
  output_show = tf.concat(feature_list, axis=0)

  tf.summary.image(str(keys) + '_features', output_show, 1000)

merge_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./visual_logs')

with tf.Session() as sess:
  saver.restore(sess,'./cifar_10_progress/')
  summ = sess.run(merge_summary_op)
  summary_writer.add_summary(summ)
