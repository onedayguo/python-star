import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt

mnist = read_data_sets('MNIST_data', one_hot=True)
train_img = mnist.train.images
validation_img = mnist.validation.images
test_img = mnist.test.images
batch_size = 100

X_holder = tf.placeholder(tf.float32)
y_holder = tf.placeholder(tf.float32)
print(dir(mnist)[-10:])
print(mnist.train.num_examples)
print(mnist.validation.num_examples)
print(mnist.test.num_examples)
images = mnist.train.images
print('images数据类型：', type(images), '数据维度：', images.shape)

# image = mnist.train.images[3].reshape(28, 28)
# plt.subplot(131)
# plt.imshow(image)
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(image, cmap='gray_r')
# plt.axis('off')
# plt.show()

# import math
# import numpy as np
#
#
# def drawDigit(position, image, title):
#     plt.subplot(*position)
#     plt.imshow(image.reshape(-1, 28), cmap='gray_r')
#     plt.axis('off')
#     plt.title(title)
#
#
# def batchDraw(batch_size):
#     images, labels = mnist.train.next_batch(batch_size)
#     image_number = images.shape[0]
#     row_number = math.ceil(image_number ** 0.5)
#     column_number = row_number
#     plt.figure(figsize=(row_number, column_number))
#     for i in range(row_number):
#         for j in range(column_number):
#             index = i * column_number + j
#             if index < image_number:
#                 position = (row_number, column_number, index + 1)
#                 image = images[index]
#                 title = 'real:%d' % (np.argmax(labels[index]))
#                 drawDigit(position, image, title)
#
#
# batchDraw(196)
# plt.show()


# Weights = tf.Variable(tf.zeros([784, 10]))
# biases = tf.Variable(tf.zeros([1, 10]))
# predict_y = tf.nn.softmax(tf.matmul(X_holder, Weights) + biases)
# loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# session = tf.Session()
# init = tf.global_variables_initializer()
# session.run(init)

# for i in range(500):
#     images, labels = mnist.train.next_batch(batch_size)
#     session.run(train, feed_dict={X_holder:images, y_holder:labels})
#     if i % 25 == 0:
#         correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         accuracy_value = session.run(accuracy, feed_dict={X_holder:mnist.test.images, y_holder:mnist.test.labels})
#         print('step:%d accuracy:%.4f' %(i, accuracy_value))