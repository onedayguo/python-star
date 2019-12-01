import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
# from tensorflow.data import Dataset
from tensorflow.keras import layers
# print(tf.__version__)
# print(tf.keras.__version__)

mnist = read_data_sets('mnist_data1/', one_hot=True)
print('训练数据数量',mnist.train.num_examples)
print('验证数据数量',mnist.validation.num_examples)
print('测试数据数量',mnist.test.num_examples)
images = mnist.train.images
print(type(images), images.shape)
