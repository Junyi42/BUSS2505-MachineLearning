from tensorflow.keras.datasets.mnist import load_data
from matplotlib import pyplot as plt
import numpy as np
from numpy import asarray
from numpy import unique
from numpy import argmax
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import random
import os

#####################################
##path need to be modified
path='/home/junyi/ML_HW4'
#####################################

#print(tf.config.list_physical_devices('GPU'),"GPU")
# load dataset
(x_train, y_train), (x_test, y_test) = load_data()

'''
# Explore the dataset
# summarize loaded dataset
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
# plot first few images
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.savefig("/home/junyi/ML_HW4/pics/mnist.jpg")
'''

# fix the random seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(seed=42)

# reshape data to have a single channel
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))#(60000,28,28,1)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# determine the shape of the input images
in_shape = x_train.shape[1:]

# determine the number of classes
n_classes = len(unique(y_train))
#print(in_shape, n_classes)#(28,28,1) (10)

# normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


def mod1():
    # define model
    model = Sequential()

    # # Convolution layer with 8 3 by 3 filters, the activation is relu
    # model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=in_shape))
    # # Max pooling layer with 2 by 2 pooling window.
    # model.add(MaxPool2D(pool_size=(2, 2)))

    # # Flatten layer
    model.add(Flatten())

    # # First hidden layer with 100 hidden nodes
    model.add(Dense(units=100, activation='sigmoid'))

    # # The output layer with 10 classes output.
    # # Use the softmax activation function for classification
    model.add(Dense(units=n_classes, activation='softmax'))

    # define loss function and optimizer
    # set the optimizer to 'sgd', then you may switch to 'adam'.
    # use cross entropy as the loss for multi-class classification
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2, validation_data=(x_test, y_test))

    # evaluate the model on training set and test set

    loss, acc1 = model.evaluate(x_train, y_train, verbose=0)
    print('Model 1: Test Accuracy on the training set: %.3f' % acc1)
    loss, acc2 = model.evaluate(x_test, y_test, verbose=0)
    print('Model 1: Test Accuracy on the test set: %.3f' % acc2)
    with open(path+'/mnist.txt','a') as f: #a追加写 w只能写
        f.write('Model 1: Test Accuracy on the training set: %.3f\n' % acc1)
        f.write('Model 1: Test Accuracy on the test set: %.3f\n\n' % acc2)


def mod2(act_func):
    # define model
    model = Sequential()

    # # Convolution layer with 8 3 by 3 filters, the activation is relu
    # model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=in_shape))
    # # Max pooling layer with 2 by 2 pooling window.
    # model.add(MaxPool2D(pool_size=(2, 2)))

    # # Flatten layer
    model.add(Flatten())
    if(act_func=="relu"):
        # # First hidden layer with 100 hidden nodes
        model.add(Dense(units=100, activation='relu'))
    elif(act_func=="tanh"):
        # # First hidden layer with 100 hidden nodes
        model.add(Dense(units=100, activation='tanh'))
    else:
        model.add(Dense(units=100, activation='sigmoid'))
    # # The output layer with 10 classes output.
    # # Use the softmax activation function for classification
    model.add(Dense(units=n_classes, activation='softmax'))

    # define loss function and optimizer
    # set the optimizer to 'sgd', then you may switch to 'adam'.
    # use cross entropy as the loss for multi-class classification
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2, validation_data=(x_test, y_test))

    # evaluate the model on training set and test set

    loss, acc1 = model.evaluate(x_train, y_train, verbose=0)
    print('Model 2 '+act_func+': Test Accuracy on the training set: %.3f' % acc1)
    loss, acc2 = model.evaluate(x_test, y_test, verbose=0)
    print('Model 2 '+act_func+': Test Accuracy on the test set: %.3f' % acc2)
    with open(path+'/mnist.txt','a') as f: #a追加写 w只能写
            f.write('Model 2 '+act_func+': Test Accuracy on the training set: %.3f\n' % acc1)
            f.write('Model 2 '+act_func+': Test Accuracy on the test set: %.3f\n\n' % acc2)

def mod3():
    # define model
    model = Sequential()

    # # Convolution layer with 8 3 by 3 filters, the activation is relu
    # model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=in_shape))
    # # Max pooling layer with 2 by 2 pooling window.
    # model.add(MaxPool2D(pool_size=(2, 2)))

    # # Flatten layer
    model.add(Flatten())

    # # First hidden layer with 100 hidden nodes
    model.add(Dense(units=100, activation='relu'))

    # # The output layer with 10 classes output.
    # # Use the softmax activation function for classification
    model.add(Dense(units=n_classes, activation='softmax'))

    # define loss function and optimizer
    # set the optimizer to 'sgd', then you may switch to 'adam'.
    # use cross entropy as the loss for multi-class classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2, validation_data=(x_test, y_test))

    # evaluate the model on training set and test set

    loss, acc1 = model.evaluate(x_train, y_train, verbose=0)
    print('Model 3: Test Accuracy on the training set: %.3f' % acc1)
    loss, acc2 = model.evaluate(x_test, y_test, verbose=0)
    print('Model 3: Test Accuracy on the test set: %.3f' % acc2)
    with open(path+'/mnist.txt','a') as f: #a追加写 w只能写
        f.write('Model 3: Test Accuracy on the training set: %.3f\n' % acc1)
        f.write('Model 3: Test Accuracy on the test set: %.3f\n\n' % acc2)

def mod4():
    # define model
    model = Sequential()

    # # Convolution layer with 8 3 by 3 filters, the activation is relu
    # model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=in_shape))
    # # Max pooling layer with 2 by 2 pooling window.
    # model.add(MaxPool2D(pool_size=(2, 2)))

    # # Flatten layer
    model.add(Flatten())

    # # First hidden layer with 100 hidden nodes
    model.add(Dense(units=128, activation='relu'))

    # # The output layer with 10 classes output.
    # # Use the softmax activation function for classification
    model.add(Dense(units=n_classes, activation='softmax'))

    # define loss function and optimizer
    # set the optimizer to 'sgd', then you may switch to 'adam'.
    # use cross entropy as the loss for multi-class classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2, validation_data=(x_test, y_test))

    # evaluate the model on training set and test set

    loss, acc1 = model.evaluate(x_train, y_train, verbose=0)
    print('Model 4: Test Accuracy on the training set: %.3f' % acc1)
    loss, acc2 = model.evaluate(x_test, y_test, verbose=0)
    print('Model 4: Test Accuracy on the test set: %.3f' % acc2)
    with open(path+'/mnist.txt','a') as f: #a追加写 w只能写
        f.write('Model 4: Test Accuracy on the training set: %.3f\n' % acc1)
        f.write('Model 4: Test Accuracy on the test set: %.3f\n\n' % acc2)

def mod5():
    # define model
    model = Sequential()

    # # Convolution layer with 8 3 by 3 filters, the activation is relu
    # model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=in_shape))
    # # Max pooling layer with 2 by 2 pooling window.
    # model.add(MaxPool2D(pool_size=(2, 2)))

    # # Flatten layer
    model.add(Flatten())

    # # First hidden layer with 100 hidden nodes
    model.add(Dense(units=100, activation='relu'))
    # second hidden layer
    model.add(Dense(units=100, activation='relu'))
    # # The output layer with 10 classes output.
    # # Use the softmax activation function for classification
    model.add(Dense(units=n_classes, activation='softmax'))

    # define loss function and optimizer
    # set the optimizer to 'sgd', then you may switch to 'adam'.
    # use cross entropy as the loss for multi-class classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2, validation_data=(x_test, y_test))

    # evaluate the model on training set and test set

    loss, acc1 = model.evaluate(x_train, y_train, verbose=0)
    print('Model 5: Test Accuracy on the training set: %.3f' % acc1)
    loss, acc2 = model.evaluate(x_test, y_test, verbose=0)
    print('Model 5: Test Accuracy on the test set: %.3f' % acc2)
    with open(path+'/mnist.txt','a') as f: #a追加写 w只能写
        f.write('Model 5: Test Accuracy on the training set: %.3f\n' % acc1)
        f.write('Model 5: Test Accuracy on the test set: %.3f\n\n' % acc2)

def mod6():
    # define model
    model = Sequential()

    # # Convolution layer with 8 3 by 3 filters, the activation is relu
    model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=in_shape))
    # # Max pooling layer with 2 by 2 pooling window.
    model.add(MaxPool2D(pool_size=(2, 2)))

    # # Flatten layer
    model.add(Flatten())

    # # First hidden layer with 100 hidden nodes
    model.add(Dense(units=100, activation='relu'))

    # # The output layer with 10 classes output.
    # # Use the softmax activation function for classification
    model.add(Dense(units=n_classes, activation='softmax'))

    # define loss function and optimizer
    # set the optimizer to 'sgd', then you may switch to 'adam'.
    # use cross entropy as the loss for multi-class classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2, validation_data=(x_test, y_test))

    # evaluate the model on training set and test set

    loss, acc1 = model.evaluate(x_train, y_train, verbose=0)
    print('Model 6: Test Accuracy on the training set: %.3f' % acc1)
    loss, acc2 = model.evaluate(x_test, y_test, verbose=0)
    print('Model 6: Test Accuracy on the test set: %.3f' % acc2)
    with open(path+'/mnist.txt','a') as f: #a追加写 w只能写
        f.write('Model 6: Test Accuracy on the training set: %.3f\n' % acc1)
        f.write('Model 6: Test Accuracy on the test set: %.3f\n\n' % acc2)

def model7(lr=1e-3):
    # define model
    model = Sequential()
    

    # # Conv1
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu',strides = 1, padding = "same", input_shape=in_shape))
    # # Max pool1
    model.add(MaxPool2D(pool_size=(2, 2),strides=2,padding='same'))
    ##conv2
    model.add(Conv2D(filters=64,kernel_size=5,strides=1,padding='same',activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2, 2),strides=2,padding='same'))

    # # Flatten layer
    model.add(Flatten())

    # # First hidden layer with 1024 hidden nodes
    model.add(Dense(units=1024, activation='relu'))

    ##dropout layer
    model.add(Dropout(0.5))
    # # The output layer with 10 classes output.
    # # Use the softmax activation function for classification
    model.add(Dense(units=n_classes, activation='softmax'))

    # define loss function and optimizer
    # set the optimizer to 'sgd', then you may switch to 'adam'.
    # use cross entropy as the loss for multi-class classification
    adam=Adam(lr=lr)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2, validation_data=(x_test, y_test))

    # evaluate the model on training set and test set

    loss, acc1 = model.evaluate(x_train, y_train, verbose=0)
    print('Model 7 with lr of '+str(lr)+': Test Accuracy on the training set: %.3f' % acc1)
    loss, acc2 = model.evaluate(x_test, y_test, verbose=0)
    print('Model 7 with lr of '+str(lr)+': Test Accuracy on the test set: %.3f' % acc2)
    with open(path+'/mnist.txt','a') as f: #a追加写 w只能写
        f.write('Model 7 with lr of '+str(lr)+': Test Accuracy on the training set: %.3f\n' % acc1)
        f.write('Model 7 with lr of '+str(lr)+': Test Accuracy on the training set: %.3f\n' % acc2)
    # model.summary()

    # make a prediction
    # image = x_train[0]
    # yhat = model.predict(asarray([image]))
    # print('Predicted: class=%d' % argmax(yhat))
def mod7():
    lrs=[1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2]
    for lr in lrs:
        model7(lr)

mod1()
mod2("relu")
mod2("tanh")
mod3()
mod4()
mod5()
mod6()
mod7()
