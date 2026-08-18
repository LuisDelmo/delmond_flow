import numpy as np

class Neuron:

    def __init__(self,activation,parents):
        self.inputs = None
        self.weigths = None
        self.bias = 0
        self.output = None
        self.activation = activation
        self.parents = np.array(*parents)

    def insert_inputs(self,inputs):
        self.inputs = inputs

    def calculate_z(self):
        self.z = self.inputs @ self.weigths + self.bias
        return self.z

    def get_inputs(self):
        return self.inputs

    def get_weights(self):
        return self.weigths

    def get_bias(self):
        return self.bias