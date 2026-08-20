import numpy as np


#TODO maybe class methods for activation and make it for layers
def linear(z):
    return z

def relu(z):
    return np.maximum(0.001 * z, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))