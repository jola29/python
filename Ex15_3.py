#Christian , Nina , Joshua , Armin 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



(x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

for i in [1239,1314,1326,22062,22090,23136,37059,37118,43089,43136]:
    plt.imshow(x_train[i], cmap = plt.get_cmap('Greys'))
    plt.show()
    print(y_train[i])


#Christian: 70%
#Armin: 70%
#Nina: 60%
#Joshua: 60%
