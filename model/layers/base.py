from ..neuron import Neuron
import numpy as np
from ..activations import Activations,Relu,Sigmoid

class Layer:
    activations = {'relu': Relu,'linear': ...,'sigmoid': Sigmoid}

    def __init__(self,units,activation='linear'):
        self.activation = activation
        self.parse_activation()

        self.units = units
        self.activation = activation
        self.output_matrix = None
        self.input_values = False
        self.layer_weights = None
        self.layer_bias = None

    def parse_activation(self):
        if self.activation in self.activations:
            self.activation = self.activations[self.activation]
        if self.activation in self.activations.keys():
            return True
        raise ValueError(f'Invalid activation: {self.activation}')
                

    def create_neurons(self,parent_layer=None):
        if parent_layer is None:
            raise ValueError('No previous/input neuron')
            
        self.units = np.array(
                        Neuron(activation=self.activation,
                                parents=parent_layer.get_neurons())
                                *self.units)

    def get_neurons(self):
        return self.units

    def get_neurons_weights(self):
        self.layer_weights = np.array(
            neuron.get_weights() 
            for neuron in self.get_neurons())

    def fit_layer(self,input_value):
        shape_input = input_value.shape
        if shape_input >= 2:
            self.features_shape = shape_input[1]
        else:
            self.features_shape = shape_input[0]

        self.input_values = input_value
        
        self.layer_weights = np.zeros(self.units.self.features_shape)

    def calculate_Z(self):
        if not self.input_values:
            raise ValueError('Data not yet fit to the layer')
        #TODO maybe add a verifier to Transpose or mult or not
        self.z = self.input_values @ self.layer_weights.T + self.layer_bias


    def foward_pass_layer(self):
        ...

        