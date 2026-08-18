import numpy as np
from .base import Model
from .layers import Layer



class Sequential(Model):

    def __init__(self,layers):
        try:
            for layer in layers:
                if not isinstance(layer,Layer): #TODO make layer later on Dense
                    raise ValueError(f'{layer} is not a layer')
        except:
            raise ValueError('Invalid input in sequential')
        self.layers = np.array(layers)
        
    #TODO Cuda maybe later?
    def fit(self,X_train,y_train,epochs):
        
        for layer in self.layers:
            for neuron in layer.units:
                neuron.insert_inputs(X_train)
        
    