import numpy as np

class Losses:
    def __init__(self,y_train,y_hat):
        self.y_train = y_train
        self.y_hat = y_hat
        self.loss = None

class Meansquarederror(Losses):
    def __init__(self,y_train,y_hat):
        self.loss = np.mean((y_train-y_hat)**2)
        return self.loss


class SparseCategoricalCrossEntropy(Losses):
    def __init__(self,y_train,y_hat):
        super().__init__(y_train,y_hat)
        
        hat_shape = y_hat.shape
        m = y_train.shape

        #search for each y = y_hat
        y_j_hat = y_hat[np.arange(m),y_train]

        self.loss = -np.log(y_j_hat)

        return self.loss

    


y_train = np.array([
    0, 1, 2, 3, 4,
    5, 6, 7, 8, 9
])


y_hat_high_loss = np.array([
    [.10, .91, .01, .01, .01, .01, .01, .01, .01, .01], # correct 0 gets .01
    [.91, .10, .01, .01, .01, .01, .01, .01, .01, .01], # correct 1 gets .01
    [.91, .01, .01, .01, .01, .01, .01, .01, .01, .01],
    [.91, .01, .01, .01, .01, .01, .01, .01, .01, .01],
    [.91, .01, .01, .01, .01, .01, .01, .01, .01, .01],
    [.91, .01, .01, .01, .01, .01, .01, .01, .01, .01],
    [.91, .01, .01, .01, .01, .01, .01, .01, .01, .01],
    [.91, .01, .01, .01, .01, .01, .01, .01, .01, .01],
    [.91, .01, .01, .01, .01, .01, .01, .01, .01, .01],
    [.91, .01, .01, .01, .01, .01, .01, .01, .01, .01],
])










#softmax example n = 10 categories === e^z/ for i in range(n): =+ e^zn or - log(an) if y_train = n