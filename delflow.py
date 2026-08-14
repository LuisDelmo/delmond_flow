import numpy as np
import multiprocessing
from losses import Meansquarederror

class neuron:

    def __init__(self,activation):
        self.inputs = None
        self.weigths = None
        self.bias = 0
        self.output = None
        self.activation = activation

    def insert_inputs(self,inputs):
        self.inputs = inputs

    def calculate_z(self):
        self.output = self.inputs @ self.weigths + self.bias
        return self.output

    def get_weights(self):
        return self.weigths


class Dense():
    activations = {'relu','linear','softmax'}
    
    def __init__(self,units,activation='linear'):
            self.units = np.array(neuron())
            self.activation = activation
            self.output_matrix = None
            self.X_train = None
            self.y_train = None
    


class Sequential:

    def __init__(self,*layers):
        for layer in layers:
            if not isinstance(layer,Dense): #TODO make layer later on Dense
                raise ValueError(f'{layer} is not a layer')
        self.layers = np.array(layers)


    def compile(self,optmizer=None,loss_func=Meansquarederror):
        self.loss = loss_func
        self.optmizer = optmizer
        
    def fit(self,X_train,y_train):
        processes = np.array()

        for layer in self.layers:
            for neuron in layer.units:
                neuron.insert_inputs(X_train)



        


X = np.array([
    [50, 1, 20],
    [70, 2, 10],
    [90, 3, 5],
    [120, 3, 2],
    [150, 4, 1]
], dtype=float)

# Price in thousands
y = np.array([
    150,
    220,
    300,
    410,
    520
], dtype=float)


hidden_weights = np.array([
    [ 0.69935616, -0.04220045, -0.04205813],  # neuron 1
    [-0.57914705,  1.23070657,  0.29092043],  # neuron 2
    [-0.63639420,  1.34085953,  0.31091339],  # neuron 3
    [ 0.97371931, -2.05799811, -0.48059770],  # neuron 4
    [ 0.44058893, -0.22629803,  0.63683221],  # neuron 5
])


hidden_biases = np.array([
    -0.02202153,   # neuron 1
    -0.58022189,   # neuron 2
    -0.63159202,   # neuron 3
     0.96970518,   # neuron 4
     1.17776076,   # neuron 5
])