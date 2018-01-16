######################################################################
#Neural Style Transformation Based on VGG_BN, Resnet-20 and Resnet-56#
#																																		 #
#						Zhanlue Yang, Xiaoxiang Zhang, Shuizhou Wang						 #
######################################################################

0. Programming Language and Libs:
*This project is written in Python2, please verify your version of python (especially when using jupyter notebook)
*Libs used: cPickle, Tensorflow, numpy, PIL, scipy


1. Brief Intro
This project is developed with reference to the following github projects:
(1)Resnet20, Resnet56: https://github.com/xuyuwei/resnet-tf
(2)VGG19: https://github.com/machrisaa/tensorflow-vgg
(3)Neural-Style: https://github.com/anishathalye/neural-style 

We further implemented Neural-Style Transformation on Resnet-20 and Resnet-56 networks, as well as VGG_BN network
***Important

2. Documents
-model: pre-trained models of resnet-20, resnet-56, and VGG_BN, cifar10 dataset was used for training
-renet: neural-style transfer code for resnet-20 network, need to run training code in resnet_train first
-resnet_train: training codes for resnet-20, using cifar10 dataset.
-resnet56: neural-style transfer code for resnet-56
-resnet56_train: training codes for resnet-56, using cifar10 dataset
-vgg_trans: training and neural-style transfer code for vgg_bn network

3. How to Use
We support neural-style transfer for 3 different networks.
(1)VGG_BN
-Use Jupyter Notebook
cd vgg_trans
run vgg_bn_Script.ipynb

(2)Resnet-20 & Resnet-56
cd resnet_train (res-20) or resnet56_train (res-56)
run resnet20_Script.ipynb (res-20) or Script.ipynb (res-56)

4. Tensorboard Support for Feature Map
In resnet, resnet56, and vgg_trans, we have image_show.py document for extracting feature maps.
Simply Run:
				python image_show.py
Then:
				tensorboard --logdir=visual_log
And then go to tensorboard running page to obtain the feature maps
