import numpy as np
from quantumbrain.graph import graph


class Optimizer:
    def apply_gradients(self):
        pass


class SDG(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def apply_gradients(self):
        for key in graph.params.keys():
            graph.params[key] -= self.lr * graph.grads[key]


class Momentum(Optimizer):
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def apply_gradients(self):
        if self.v is None:
            self.v = {}
            for key, val in graph.params.items():
                self.v[key] = np.zeros_like(val)

        for key in graph.params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * graph.grads[key]
            graph.params[key] += self.v[key]


class Nesterov(Optimizer):
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def apply_gradients(self):
        if self.v is None:
            self.v = {}
            for key, val in graph.params.items():
                self.v[key] = np.zeros_like(val)

        for key in graph.params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * graph.grads[key]

            graph.params[key] += self.momentum * self.momentum * self.v[key]
            graph.params[key] -= (1 + self.momentum) * self.lr * graph.grads[key]


class AdaGrad(Optimizer):
    def __init__(self, lr=1e-4):
        self.lr = lr
        self.delta = 1e-7
        self.h = None

    def apply_gradients(self):
        if self.h is None:
            self.h = {}
            for key, val in graph.params.items():
                self.h[key] = np.zeros_like(val)

        for key in graph.params.keys():
            self.h[key] += graph.grads[key] * graph.grads[key]
            graph.params[key] -= self.lr * graph.grads[key] / (np.sqrt(self.h[key] + self.delta))


class RMSprop(Optimizer):
    def __init__(self, lr=0.001, rho=0.9):
        self.lr = lr
        self.rho = rho
        self.delta = 1e-7
        self.h = None

    def apply_gradients(self):
        if self.h is None:
            self.h = {}
            for key, val in graph.params.items():
                self.h[key] = np.zeros_like(val)

        for key in graph.params.keys():
            self.h[key] *= self.rho
            self.h[key] += (1 - self.rho) * graph.grads[key] * graph.grads[key]
            graph.params[key] -= self.lr * graph.grads[key] / (np.sqrt(self.h[key] + self.delta))


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.delta = 1e-7
        self.m = None
        self.v = None
        self.t = 0

    def apply_gradients(self):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in graph.params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.t += 1
        lr = self.lr * np.sqrt(1.0 - self.beta_2**self.t) / (1.0 - self.beta_1**self.t)

        for key in graph.params.keys():
            self.m[key] += (1.0 - self.beta_1) * (graph.grads[key] - self.m[key])
            self.v[key] += (1.0 - self.beta_2) * (graph.grads[key]**2 - self.v[key])

            graph.params[key] -= lr * self.m[key] / (np.sqrt(self.v[key]) + self.delta)