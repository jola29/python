
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



(x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape)

#x_train = tf.cast(x_train.reshape(60000,784),dtype=tf.float32)
#x_test = tf.cast(x_test.reshape(10000,784),dtype=tf.float32)

#y_train_softmax = np.zeros(shape=(60000,10), dtype = np.float32)
#y_test_softmax = np.zeros(shape=(10000,10), dtype = np.float32)

#for i in range(10):
#	y_train_softmax[np.where(y_train == i),i] = 1
#	y_test_softmax[np.where(y_test == i),i] = 1

#y_train_softmax = tf.convert_to_tensor(y_train_softmax)
#y_test_softmax = tf.convert_to_tensor(y_test_softmax)


plt.imshow(x_train[12500], cmap = plt.get_cmap('Greys'))

plt.show()