import numpy as np



class Meansquarederror:
    def __init__(self,y_train,y_hat):
        self.loss = np.mean((y_train-y_hat)**2)
        return self.loss