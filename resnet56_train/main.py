import models_train
import cPickle
import numpy as np
import tensorflow as tf
import os

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.datasets.cifar import load_batch
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file
epoches = 30
dataset = 'cifar10'

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def one_hot_vec_10(label):
    vec = np.zeros(10)
    vec[label] = 1
    return vec

def one_hot_vec_100(label):
    vec = np.zeros(100)
    vec[label] = 1
    return vec

def load_cifar100_data(label_mode='fine'):
  """loads cifar100 dataset.
  arguments:
      label_mode: one of "fine", "coarse".
  returns:
      tuple of numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  raises:
      valueerror: in case of invalid `label_mode`.
  """
  if label_mode not in ['fine', 'coarse']:
    raise valueerror('label_mode must be one of "fine" "coarse".')
  print ('load cifar100 dataset')
  dirname = 'cifar-100-python'
  origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
  path = get_file(dirname, origin=origin, untar=True)

  fpath = os.path.join(path, 'train')
  x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

  fpath = os.path.join(path, 'test')
  x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  if K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

  y_test = map(one_hot_vec_100, y_test)
  y_train = map(one_hot_vec_100, y_train)
  return (x_train, y_train, x_test, y_test)


def load_cifar10_data():
    print ('load cifar10 dataset')
    x_all = []
    y_all = []
    for i in range (5):
        d = unpickle("cifar-10-batches-py/data_batch_" + str(i+1))
        x_ = d['data']
        y_ = d['labels']
        x_all.append(x_)
        y_all.append(y_)

    d = unpickle('cifar-10-batches-py/test_batch')
    x_all.append(d['data'])
    y_all.append(d['labels'])

    x = np.concatenate(x_all) / np.float32(255)
    y = np.concatenate(y_all)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))
    
    pixel_mean = np.mean(x[0:50000],axis=0)
    x -= pixel_mean

    y = map(one_hot_vec_10, y)
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return (X_train, Y_train, X_test, Y_test)

def main():
    if dataset == 'cifar10':
        directory = '../resnet56/cifar_10_progress/'
        cifar = 10
        classes = 10
        load_function = load_cifar10_data()
    elif dataset == 'cifar100':
        directory = '../resnet56/cifar_100_progress/'
        cifar = 100
        classes = 100
        load_function = load_cifar100_data()

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
    flags.DEFINE_integer('batch_size', 25, 'Batch size')

    X_train, Y_train, X_test, Y_test = load_function

    Y_train = np.array(Y_train)

    batch_size = 128

    X = tf.placeholder("float", [batch_size, 32, 32, 3])
    Y = tf.placeholder("float", [batch_size, classes])
    learning_rate = tf.placeholder("float", [])

    # ResNet Models
    net,saver = models.resnet(X, 56, cifar)
    # net = models.resnet(X, 32)
    # net = models.resnet(X, 44)
    # net = models.resnet(X, 56)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=Y, name='xentropy'),name='loss')
    #cross_entropy = -tf.reduce_sum(Y*tf.log(net))
    opt = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999, 1e-08)
    train_op = opt.minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    best_acc = 0
    for j in range (epoches):
        c = np.c_[X_train.reshape(len(X_train), -1), Y_train.reshape(len(Y_train), -1)]
        np.random.shuffle(c)
        X_train = c[:, :X_train.size//len(X_train)].reshape(X_train.shape)  
        Y_train = c[:, X_train.size//len(X_train):].reshape(Y_train.shape)
        
        final = int(50000 / batch_size) * batch_size    
        for i in range (0, final, batch_size):
            feed_dict={
                X : X_train[i:i + batch_size],
                Y : Y_train[i:i + batch_size],
                learning_rate: 0.001}
            _,out_loss = sess.run([train_op, loss], feed_dict=feed_dict)
            print ('loss: ', out_loss)

        cum_acc = 0
        counts = 0
        for i in range (0, 10000, batch_size):
            if i + batch_size < 10000:
                acc = sess.run([accuracy],feed_dict={
                    X: X_test[i:i+batch_size],
                    Y: Y_test[i:i+batch_size]
                })
                counts += 1
                cum_acc += acc[0]
        print (cum_acc / counts)
        acc_now = cum_acc / counts
        if (acc_now >= best_acc):
            best_acc = acc_now
            print ('Reached Best Accuracy: ', best_acc)
            saver.save(sess, directory)

    sess.close()

if __name__ == '__main__':
    main()
