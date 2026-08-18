import numpy as np
from .base import Activations

class Sigmoid(Activations):

    def calculate(self):
        self.output = 1/(1+np.exp(self.z))
        return self.output
