# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 12:52:39 2020

@author: User
"""
import os
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dense, UpSampling2D, Conv2D, Flatten
from keras.layers import Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np


class DCGAN(object):
    """ Deep Convolutional Generative Adversarial Network class """
    def __init__(self, width=28, height=28, channels=1):

        self.width = width
        self.height = height
        self.channels = channels
        self.latent_dim = 10
        
        self.shape = (self.width, self.height, self.channels)

        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
        
        #self.D_optim = rmsprop(lr=0.0004)
        #self.G_optim = rmsprop(lr=0.0002)
        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)


    def __generator(self):
        """ Declare generator """
        
        model = Sequential()
        model.add(Dense(128 * 7 * 7, input_dim=self.latent_dim, activation="relu"))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.channels, kernel_size=4, padding='same')) 
        model.add(Activation('tanh'))
        
        #model.summary()
        return model

    def __discriminator(self):
        """ Declare discriminator """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=4,strides=2,input_shape=self.shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64,kernel_size=4, strides=2,padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(1,activation='sigmoid'))
        
        #model.summary()
        return model

    def __stacked_generator_discriminator(self):

        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)
        
        #model.summary()
        return model

    def train(self, X_train, epochs=10000, batch = 128, save_interval = 500):
        
        for cnt in range(epochs):
          
            random_index = np.random.randint(0, len(X_train) - np.int64(batch/2))
            legit_images = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels)

            gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 10))
            syntetic_images = self.G.predict(gen_noise)
            
            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))
                       
            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)
            
            # train generator
            noise = np.random.normal(0, 1, (batch, 10))
            y_mislabled = np.ones((batch, 1))

            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

            print ('epoch: %d, [Discriminator :: d_loss: %f, accuracy: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0],d_loss[1], g_loss))

            if cnt % save_interval == 0:
                self.__plot_images(save2file=True, step=cnt)
        
        #save weight and model
        if not os.path.exists("./saved_model_weights"):
            os.makedirs('saved_model_weights', exist_ok=True)
            self.G.save_weights('saved_model_weights/generator_weights.h5')
            self.D.save_weights('saved_model_weights/discriminator_weights.h5')
            self.stacked_generator_discriminator.save_weights('saved_model_weights/combined_weights.h5')
                

    def __plot_images(self, save2file=False, samples=16, step=0):
        ''' Plot and generated images '''
        if not os.path.exists("./images"):
            os.makedirs("./images")
        filename = "./images/mnist_%d.png" % step
        noise = np.random.normal(0, 1, (samples, 10))

        images = self.G.predict(noise)

        plt.figure(figsize=(10, 10))

        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.height, self.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()
            
     
if __name__ == '__main__':
    
    
    (X_train, _), (_, _) = mnist.load_data()

    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],X_train.shape[2],1)).astype('float32') / 255.0

    dcgan = DCGAN()
    dcgan.train(X_train)

