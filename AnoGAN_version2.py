# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 05:46:37 2020

@author: User
"""

from DCGAN import DCGAN
from keras import backend as K
from keras.datasets import mnist
from keras import initializers
import tensorflow as tf
from keras.models import Model
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.layers import Input, Reshape, Dense, UpSampling2D, Conv2D, Flatten
from keras.layers import Activation, Dropout

class AnoGAN(DCGAN):
    
    def __init__(self, data_x, width=28, height=28, channels=1):
        super(AnoGAN, self).__init__(width=28, height=28, channels=1)
        self.intermidiate_model = self.__feature_extractor(data_x)
        self.ano_model = self.__anamoly_detector()

        
    def sum_of_residual(self, y_true, y_pred):
        return tf.reduce_sum(abs(y_true - y_pred))


    def __feature_extractor(self, data_x):  
        self.__load_weight(data_x)
        intermidiate_model = Model(inputs=self.D.layers[0].input, outputs=self.D.layers[-4].output)
        intermidiate_model.trainable = False
        intermidiate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        #print("YES")
        return intermidiate_model


    def __load_weight(self, data_x):
        if not os.path.exists("./saved_model_weights"):
            self.train(data_x)
        else:
            self.G.load_weights('saved_model_weights/generator_weights.h5')
            self.D.load_weights('saved_model_weights/discriminator_weights.h5')
        
            
    def __anamoly_detector(self):    
        
        g = Model(inputs=self.G.layers[0].input, outputs=self.G.layers[-1].output)
        g.trainable = False
        
        # Input layer cann't be trained. Add new layer as same size & same distribution
        aInput = Input(shape=(10,))
    
        # G & D feature
        G_out = g(aInput)
        self.intermidiate_model.trainable = False
        D_out= self.intermidiate_model(G_out)    
        model = Model(inputs=aInput, outputs=[G_out, D_out])
        model.compile(loss=self.sum_of_residual, loss_weights= [0.90, 0.10], optimizer='rmsprop')
    
        # batchnorm learning phase fixed (test) : make non trainable
        K.set_learning_phase(0)
    
        return model
    
    
    def compute_anomaly_score(self, x, iterations=500, d=None):
        
        z = np.random.uniform(0, 1, size=(1, 10))
    
        d_x = self.intermidiate_model.predict(x)
        
        loss = 0.90*self.sum_of_residual(x, self.ano_model.output[0]) + 0.10*self.sum_of_residual(d_x, self.ano_model.outputs[1])
        
        grads = K.gradients(loss, self.ano_model.input)[0]
        #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        
        iterate = K.function([self.ano_model.input], [self.ano_model.output, grads])
        
        for i in range(100):
            _, grads_value = iterate([z])
            #print(grads_value)
            z -= grads_value * 0.05
    
        similar_data, _ = self.ano_model.predict(z)
        d_z = self.intermidiate_model.predict(similar_data)
        
        #print(type(self.sum_of_residual(x, similar_data)))
        score = 0.90*np.sum(self.sum_of_residual(x, similar_data)) + 0.10*np.sum(self.sum_of_residual(d_x, d_z)) 
        
        
        return score, similar_data



if __name__ == '__main__':
    
    
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    train_mask = np.isin(Y_train, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    X_train, Y_train = X_train[train_mask], Y_train[train_mask]
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    
    anogan = AnoGAN(X_train, 28, 28, 1)
    for i in range(10):
        mask = np.where(Y_test==i)
        mask = mask[0][3]
        test_img = X_test[mask]
        ano_score, similar_img = anogan.compute_anomaly_score(np.reshape(test_img,(1,28,28,1)))
        
        plt.figure(figsize=(2, 2))
        plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
        residual  = test_img.reshape(28,28) - similar_img.reshape(28, 28)
        plt.imshow(residual, cmap='jet', alpha=.5)
        plt.show()
        print(i, ":anomaly score : " , ano_score)
        
            