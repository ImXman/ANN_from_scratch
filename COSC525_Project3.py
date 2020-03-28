# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:16:40 2020

@author: Yang Xu
"""
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import UpSampling2D, Lambda, Conv2DTranspose, Reshape

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import time
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

##-----------------------------------------------------------------------------
##task1
##read Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
##Min-Max scaling
x_train = x_train.reshape(60000,28*28)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
y_train = keras.utils.to_categorical(y_train, num_classes=10)

x_test = x_test.reshape(10000,28*28)
scaler = MinMaxScaler()
scaler.fit(x_test)
x_test = scaler.transform(x_test)

model = Sequential([
    Dense(784, input_shape=(784,)),
    Activation('tanh'),
    Dense(512),
    Activation('sigmoid'),
    Dense(100),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

time_callback = TimeHistory()
history = model.fit(x_train, y_train, epochs=50, batch_size=200,callbacks=[time_callback])
times = time_callback.times
times = [sum(times[:i]) for i in range(50)]

##plot epoch-loss
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set'], loc='upper right')
plt.show()

##plot time-loss
plt.plot(times,history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Time(Second)')
plt.legend(['Training set'], loc='upper right')
plt.show()

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
cm = confusion_matrix(y_test, y_pred)
accuracy = np.sum(np.diag(cm))/np.sum(cm)
np.savetxt("confusion_matrix.csv",cm,delimiter=",",fmt="%i")

##-----------------------------------------------------------------------------
##task2
##read Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
##Min-Max scaling
x_train = x_train.reshape(60000,28*28)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_train = x_train.reshape(60000,28,28,1)
y_train = keras.utils.to_categorical(y_train, num_classes=10)

x_test = x_test.reshape(10000,28*28)
scaler = MinMaxScaler()
scaler.fit(x_test)
x_test = scaler.transform(x_test)
x_test = x_test.reshape(10000,28,28,1)

model = Sequential([
    Conv2D(40, kernel_size=(5, 5),input_shape=(28,28,1),strides=(1, 1), padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(100),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

time_callback = TimeHistory()
history = model.fit(x_train, y_train, epochs=50, batch_size=200,callbacks=[time_callback])
times = time_callback.times
times = [sum(times[:i]) for i in range(50)]

##plot epoch-loss
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set'], loc='upper right')
plt.show()

##plot time-loss
plt.plot(times,history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Time(Second)')
plt.legend(['Training set'], loc='upper right')
plt.show()

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
cm = confusion_matrix(y_test, y_pred)
accuracy = np.sum(np.diag(cm))/np.sum(cm)
np.savetxt("confusion_matrix.csv",cm,delimiter=",",fmt="%i")

##-----------------------------------------------------------------------------
##task3
##read Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
##Min-Max scaling
x_train = x_train.reshape(60000,28*28)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_train = x_train.reshape(60000,28,28,1)
y_train = keras.utils.to_categorical(y_train, num_classes=10)

x_test = x_test.reshape(10000,28*28)
scaler = MinMaxScaler()
scaler.fit(x_test)
x_test = scaler.transform(x_test)
x_test = x_test.reshape(10000,28,28,1)

model = Sequential([
    Conv2D(48, kernel_size=(3, 3),input_shape=(28,28,1),strides=(1, 1), padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(96, kernel_size=(3, 3),strides=(1, 1), padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(100),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

time_callback = TimeHistory()
history = model.fit(x_train, y_train, epochs=50, batch_size=200,callbacks=[time_callback])
times = time_callback.times
times = [sum(times[:i]) for i in range(50)]

##plot epoch-loss
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set'], loc='upper right')
plt.show()

##plot time-loss
plt.plot(times,history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Time(Second)')
plt.legend(['Training set'], loc='upper right')
plt.show()

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
cm = confusion_matrix(y_test, y_pred)
accuracy = np.sum(np.diag(cm))/np.sum(cm)
np.savetxt("confusion_matrix.csv",cm,delimiter=",",fmt="%i")

##-----------------------------------------------------------------------------
##task4
##read Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
##Min-Max scaling
x_train = x_train.reshape(60000,28*28)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_train = x_train.reshape(60000,28,28,1)
y_train = keras.utils.to_categorical(y_train, num_classes=10)

x_test = x_test.reshape(10000,28*28)
scaler = MinMaxScaler()
scaler.fit(x_test)
x_test = scaler.transform(x_test)
x_test = x_test.reshape(10000,28,28,1)

model = Sequential([
    ##VGG-like model
    Conv2D(48, kernel_size=(3, 3),input_shape=(28,28,1),strides=(1, 1), padding='valid'),
    Activation('relu'),
    Conv2D(96, kernel_size=(3, 3),strides=(1, 1), padding='valid'),
    Activation('relu'),
    Conv2D(128, kernel_size=(3, 3),strides=(1, 1), padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

time_callback = TimeHistory()
history = model.fit(x_train, y_train, epochs=50, batch_size=200,callbacks=[time_callback])
times = time_callback.times
times = [sum(times[:i]) for i in range(50)]

##plot epoch-loss
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set'], loc='upper right')
plt.show()

##plot time-loss
plt.plot(times,history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Time(Second)')
plt.legend(['Training set'], loc='upper right')
plt.show()

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
cm = confusion_matrix(y_test, y_pred)
accuracy = np.sum(np.diag(cm))/np.sum(cm)
np.savetxt("confusion_matrix.csv",cm,delimiter=",",fmt="%i")

##-----------------------------------------------------------------------------
##task5
##read Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
##Min-Max scaling
x_train = x_train.reshape(60000,28*28)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
y_train = keras.utils.to_categorical(y_train, num_classes=10)

x_test = x_test.reshape(10000,28*28)
scaler = MinMaxScaler()
scaler.fit(x_test)
x_test = scaler.transform(x_test)

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
    args (tensor): mean and log of variance of Q(z|X)
    # Returns
    z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

batch_size = 200
epochs = 50
input_shape = x_train.shape[1]
image_size = 28
intermediate_dim = 128
latent_dim = 10

##2-Conv2d-layer encoder
inputs = Input(shape=input_shape, name='encoder_input')
x = Reshape((28,28,1),name="reshap1e")(inputs)
x = Conv2D(24, kernel_size=(3, 3),strides=(1, 1), 
           padding='same',activation='relu', name="encoder_Conv1")(x)
x = MaxPooling2D(pool_size=(2, 2),name="encoder_MaxPooling1")(x)
x = Conv2D(48, kernel_size=(3, 3),strides=(1, 1), 
           padding='same',activation='relu', name="encoder_Conv2")(x)
x = MaxPooling2D(pool_size=(2, 2),name="encoder_MaxPooling2")(x)
x = Flatten()(x)
x = Dense(intermediate_dim,activation='relu',name="encoder_hidden")(x)

z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder_output')
encoder.summary()

##decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu', name="decoder_hidden")(latent_inputs)
x = Dense(2352, activation='relu', name="decoder_hidden2")(x)
x = Reshape((7,7,48),name="reshape2")(x)##dim of output of the second MaxPooling layer in encoder
x = UpSampling2D(name="decoder_UpSampling1")(x)
x = Conv2DTranspose(24,3,1,padding='same',activation='relu',name="decoder_deConv1")(x)
x = UpSampling2D(name="decoder_UpSampling2")(x)
x = Conv2DTranspose(1,3,1,padding='same',activation='sigmoid',name="decoder_deConv2")(x)
outputs = Reshape((input_shape,),name="reshape3")(x)

decoder = Model(latent_inputs, outputs, name='decoder_output')
decoder.summary()

##variational AE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')
vae.summary()

#setting loss
reconstruction_loss = mse(inputs, outputs)
#reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= input_shape
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
#vae_loss = reconstruction_loss+kl_loss
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

time_callback = TimeHistory()
history = vae.fit(x_train, epochs=epochs, batch_size=batch_size,callbacks=[time_callback])
times = time_callback.times
times = [sum(times[:i]) for i in range(50)]

##plot epoch-loss
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set'], loc='upper right')
plt.show()

##plot time-loss
plt.plot(times,history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Time(Second)')
plt.legend(['Training set'], loc='upper right')
plt.show()

##plot randomly generated clothes
n = 10
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates corresponding to the 2D plot
# of digit classes in the latent space
for i in range(n):
    for j in range(n):
        clothes_latent = np.random.uniform(-4,4,10).reshape(1,10)
        x_decoded = decoder.predict(clothes_latent)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
start_range = digit_size // 2
end_range = (n - 1) * digit_size + start_range + 1
pixel_range = np.arange(start_range, end_range, digit_size)
plt.imshow(figure, cmap='Greys_r')
plt.show()

##-----------------------------------------------------------------------------
##Alternative model
##3-Conv2d-layer encoder
inputs = Input(shape=input_shape, name='encoder_input')
x = Reshape((28,28,1),name="reshap1e")(inputs)
x = Conv2D(24, kernel_size=(3, 3),strides=(1, 1), 
           padding='same',activation='relu', name="encoder_Conv1")(x)
x = MaxPooling2D(pool_size=(2, 2),name="encoder_MaxPooling1")(x)
x = Conv2D(48, kernel_size=(3, 3),strides=(1, 1), 
           padding='same',activation='relu', name="encoder_Conv2")(x)
x = MaxPooling2D(pool_size=(2, 2),name="encoder_MaxPooling2")(x)
x = Conv2D(96, kernel_size=(3, 3),strides=(1, 1), 
           padding='same',activation='relu', name="encoder_Conv3")(x)
x = Flatten()(x)
x = Dense(intermediate_dim,activation='relu',name="encoder_hidden")(x)

z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder_output')
encoder.summary()

##decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu', name="decoder_hidden")(latent_inputs)
x = Dense(4704, activation='relu', name="decoder_hidden2")(x)
x = Reshape((7,7,96),name="reshape2")(x)##dim of output of the second MaxPooling layer in encoder
x = Conv2DTranspose(48,3,1,padding='same',activation='relu',name="decoder_deConv1")(x)
x = UpSampling2D(name="decoder_UpSampling2")(x)
x = Conv2DTranspose(24,3,1,padding='same',activation='relu',name="decoder_deConv2")(x)
x = UpSampling2D(name="decoder_UpSampling3")(x)
x = Conv2DTranspose(1,3,1,padding='same',activation='sigmoid',name="decoder_deConv3")(x)
outputs = Reshape((input_shape,),name="reshape3")(x)

decoder = Model(latent_inputs, outputs, name='decoder_output')
decoder.summary()

##variational AE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')
vae.summary()

#setting loss
reconstruction_loss = mse(inputs, outputs)
#reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= input_shape
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
#vae_loss = reconstruction_loss+kl_loss
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

time_callback = TimeHistory()
history = vae.fit(x_train, epochs=epochs, batch_size=batch_size,callbacks=[time_callback])
times = time_callback.times
times = [sum(times[:i]) for i in range(50)]

##plot epoch-loss
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set'], loc='upper right')
plt.show()

##plot time-loss
plt.plot(times,history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Time(Second)')
plt.legend(['Training set'], loc='upper right')
plt.show()

##plot randomly generated clothes
n = 10
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates corresponding to the 2D plot
# of digit classes in the latent space
for i in range(n):
    for j in range(n):
        clothes_latent = np.random.uniform(-4,4,10).reshape(1,10)
        x_decoded = decoder.predict(clothes_latent)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
start_range = digit_size // 2
end_range = (n - 1) * digit_size + start_range + 1
pixel_range = np.arange(start_range, end_range, digit_size)
plt.imshow(figure, cmap='Greys_r')
plt.show()
