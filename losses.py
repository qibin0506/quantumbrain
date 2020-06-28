from quantumbrain.functions import *


class Loss:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def __call__(self, *args, **kwargs):
        return self.call(args[0], args[1])

    def call(self, y_true, y_pred):
        pass

    def backward(self):
        pass


class MeanSquareError(Loss):
    def call(self, y_true, y_pred):
        self.y_pred = y_pred
        self.y_true = y_true

        loss = 0.5 * np.mean(np.sum((y_pred - y_true)**2, axis=1))
        return loss

    def backward(self):
        d_pred = self.y_pred - self.y_true
        return d_pred


class CategoricalCrossentropy(Loss):
    def call(self, y_true, y_pred):
        self.y_pred = softmax(y_pred)
        self.y_true = y_true

        loss = cross_entropy_error(self.y_true, self.y_pred)
        return loss

    def backward(self):
        batch_size = self.y_true.shape[0]
        d_pred = (self.y_pred - self.y_true) / batch_size

        return d_pred


class BinaryCrossentropy(Loss):
    def call(self, y_true, y_pred):
        self.y_pred = y_pred
        self.y_true = y_true

        loss = np.mean(-np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1))
        return loss

    def backward(self):
        d_pred = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred))
        return d_pred


MSE = MeanSquareError
BCE = BinaryCrossentropy
CE = CategoricalCrossentropy
