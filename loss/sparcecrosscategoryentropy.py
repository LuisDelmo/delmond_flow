from .base import Losses
import numpy as np


class SparseCategoricalCrossEntropy(Losses):
    def __init__(self,y_train,y_hat):
        super().__init__(y_train,y_hat)
        
        hat_shape = y_hat.shape
        m = y_train.shape

        #search for each y = y_hat
        y_j_hat = y_hat[np.arange(m),y_train]

        self.loss = -np.log(y_j_hat)

        return self.loss