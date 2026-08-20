import numpy as np

class MeanSquaredError:

    @staticmethod
    def calculate(y_train,y_hat):
        loss = np.mean((y_train-y_hat)**2)
        return loss

    @staticmethod
    def gradient(y_train,y_hat):

        n = y_train.size
        return 2/n * (y_train - y_hat)