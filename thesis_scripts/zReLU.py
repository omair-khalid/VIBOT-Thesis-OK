# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:21:33 2018

@author: Omair Khalid
"""

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import theano.tensor as T
from math import pi


class zReLU(Layer):

    def get_realpart(self,x):
        image_format = K.image_data_format()
        ndim = K.ndim(x)
        input_shape = K.shape(x)
    
        if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
            input_dim = input_shape[1] // 2
            return x[:, :input_dim]
    
        input_dim = input_shape[-1] // 2
        if ndim == 3:
            return x[:, :, :input_dim]
        elif ndim == 4:
            return x[:, :, :, :input_dim]
        elif ndim == 5:
            return x[:, :, :, :, :input_dim]    
            
    def get_imagpart(self,x):
        image_format = K.image_data_format()
        ndim = K.ndim(x)
        input_shape = K.shape(x)
    
        if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
            input_dim = input_shape[1] // 2
            return x[:, input_dim:]
    
        input_dim = input_shape[-1] // 2
        if ndim == 3:
            return x[:, :, input_dim:]
        elif ndim == 4:
            return x[:, :, :, input_dim:]
        elif ndim == 5:
            return x[:, :, :, :, input_dim:]
    
#    def get_abs(self,x):
#        real = self.get_realpart(x)
#        imag = self.get_imagpart(x)
#    
#        return K.sqrt(real * real + imag * imag)
    
    def get_angle(self,x):
        real = self.get_realpart(x)
        imag = self.get_imagpart(x)
        ang = T.arctan2(imag,real)        
       # comp_num = real + 1j*imag
        return ang #
        #T.angle(comp_num)
        
    def __init__(self, **kwargs):
#        self.output_dim = output_dim
        super(zReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
#        self.b = self.add_weight(name='b', 
#                                      shape=(input_shape[1]/2,input_shape[2],input_shape[3]),
#                                      initializer='uniform',
#                                      trainable=True)
        super(zReLU, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        real = self.get_realpart(x)
        imag = self.get_imagpart(x)        
        #mag = self.get_abs(x)
        ang = self.get_angle(x) + 0.0001
        indices1 = T.nonzero(T.ge(ang,pi/2))
        indices2 = T.nonzero(T.le(ang,0))
        
        real = T.set_subtensor(real[indices1], 0)    
        imag = T.set_subtensor(imag[indices1], 0)        
        
        real = T.set_subtensor(real[indices2], 0)    
        imag = T.set_subtensor(imag[indices2], 0)        
        
            
        act = K.concatenate([real, imag], axis=1)
        
        return act

    def compute_output_shape(self, input_shape):
        return (input_shape) 