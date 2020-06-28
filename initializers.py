import numpy as np


class NormalInitializer:
    def __call__(self, *args, **kwargs):
        return np.random.randn(*args) * 0.01


class XavierInitializer:
    def __call__(self, *args, **kwargs):
        return np.random.randn(*args) / np.sqrt(args[1])


class HeInitializer:
    def __call__(self, *args, **kwargs):
        return np.random.randn(*args) / np.sqrt(args[1] / 2)


def get(identifier):
    initializer_name = identifier.capitalize() + "Initializer"

    initializer = globals()[initializer_name]
    return initializer()

