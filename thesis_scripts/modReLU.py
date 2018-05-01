'''
Implementation of modReLU acitvation function
author: Omair Khalid
'''
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import theano.tensor as T


class modReLU(Layer):

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
    
    def get_abs(self,x):
        real = self.get_realpart(x)
        imag = self.get_imagpart(x)
    
        return K.sqrt(real * real + imag * imag)

    def __init__(self, **kwargs):
        #self.output_dim = output_dim
        super(modReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.b = self.add_weight(name='b', 
                                      shape=(input_shape[1]/2,input_shape[2],input_shape[3]),
                                      initializer='uniform',
                                      trainable=True)
        super(modReLU, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        real = self.get_realpart(x)
        imag = self.get_imagpart(x)        
        mag = self.get_abs(x)
        #ang = self.get_angle(x) 
        
        comp_num = real+ 1j*imag
        
        z_norm = mag + 0.00001
        step1 = z_norm + self.b
        step2 = K.relu(step1)
        
        
        real_act = (real/mag)*step2         
        imag_act = (imag/mag)*step2
        
        act = K.concatenate([real_act, imag_act], axis=1)

        return act

    def compute_output_shape(self, input_shape):
        return (input_shape)
