import numpy as np
from loss import Meansquarederror

class Model:
    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.activations = None
        self.model_weights = None
        self.model_biases = None


    def compile(self,optmizer=None,loss_func=Meansquarederror):
        self.loss = loss_func
        self.optmizer = optmizer

    #TODO Cuda maybe later?