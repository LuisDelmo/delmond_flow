
class Losses:
    def __init__(self,y_train,y_hat):
        self.y_train = y_train
        self.y_hat = y_hat
        self.loss = None