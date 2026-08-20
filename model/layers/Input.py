from .base import Layer

class Input(Layer):
    def __init__(self,input_shape):
        self.input_shape = input_shape

    def foward_pass_layer(self,input_value):
        self.input_value = input_value
        return self.input_value
        