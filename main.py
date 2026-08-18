from model.layers import Dense
from model import Sequential
from model.layers import Layer
import numpy as np








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





this_layer = Layer(units=5,activation='relu')







delmodel = Layer(units=5)






