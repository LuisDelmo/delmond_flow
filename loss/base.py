import numpy as np




def meansquarederror(y_train,y_hat):
    loss = np.mean((y_train-y_hat)**2)
    return loss




def sparsecrosscategoricalentropy(y_train,y_hat):

    m = y_train.shape

    #search for each y = y_hat
    y_j_hat = y_hat[np.arange(m),y_train]

    loss = -np.log(y_j_hat)

    return loss