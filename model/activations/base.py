import numpy as np
from .. import Neuron

#TODO maybe class methods for activation and make it for layers
class Activations:
    
    def __init__(self):
        self.inputs = None
        self.output = None


    def get_input(self,input):
            
        if not isinstance(input,Neuron) and not isinstance(input,Layer):
            raise ValueError('Type of input is invalid')
        
        self.z = input.calculate_z()

    

    def calculate(self):
        ...