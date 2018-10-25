# import numpy, matplotlib
import numpy as np
np.random.seed(123)

from matplotlib import pyplot as plt

# import keras modules
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils
from keras import backend as K

from keras.datasets import mnist

# load data
(x_train, y_train), (x_test,y_test) = mnist.load_data()

# pre-process steps
# x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
# x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# model
model = Sequential()
model.add(Conv2D(6, kernel_size=(6, 6),
	activation='relu',
	input_shape=input_shape))
model.add(Conv2D(12, (5, 5), activation='relu'))
model.add(Conv2D(24, (4, 4), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])

# Fit model on data
model.fit(x_train, y_train, batch_size=100, epochs=5, verbose=1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])
result = model.predict(x_test, verbose=0)
np.savetxt('Evaluation_mnist.csv', result, fmt='%.2f', delimiter=',',
	 header=" ImageID,  Label")
