import numpy as np
from .base import Activations

class Relu(Activations):

    def calculate(self):
        self.output = np.maximum(0,self.z)
        return self.output