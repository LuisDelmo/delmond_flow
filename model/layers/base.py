import numpy as np


class Layer:
    
    def __init__(self,units,activation='linear'):
        self.activation = activation
    
        self.units = units
        self.activation = activation
        self.output_matrix = None
        self.input_values = None
        self.layer_weights = None
        self.layer_bias = None
 
                
    def fit_layer(self,input_value):
        input_size = input_value.shape[-1]
        
        self.input_values = input_value
        if self.layer_weights is None:
            self.layer_weights = np.random.randn(
                                                self.units,
                                                input_size
                                            ) * 0.01

                                          
    def calculate_Z(self):
        if self.input_values is None:
            raise ValueError('Data not yet fit to the layer')
        #TODO maybe add a verifier to Transpose or mult or not
        return self.input_values @ self.layer_weights.T + self.layer_bias


    def foward_pass_layer(self,layer_input):
        self.fit_layer(layer_input)
        self.z = self.calculate_Z()
        self.output_matrix = self.activation(self.z)
        return self.output_matrix
    
    def backward_pass_layer(self,gradient):
        
        dj_dyhat = gradient
        
        dj_dz = dj_dyhat * (self.z > 0)

        n,m = self.layer_weights.shape

        