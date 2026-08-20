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
        self.layers = np.array(layers[1:])
        self.input_layer = layers[0]


    def foward_propagation(self,input_value):

        this_a = self.input_layer.foward_pass_layer(input_value)
        for layer in self.layers:
            this_a = layer.foward_pass_layer(this_a)

        return this_a
            
    def predict(self,X):

        self.output = self.foward_propagation(input_value=X)
        return self.output
        
    #TODO Cuda maybe later?
    def fit(self,X_train,y_train,epochs):
        
        
        output = self.predict(X_train)

        loss = self.loss.calculate(y_train,output)
        curr_gradient = self.loss.gradient(y_train,output)

        for layer in self.layers[::-1]:
            curr_gradient_gradient = layer.backward_pass_layer(curr_gradient)

        
    