import numpy as np


class RandomNormal:
    def __init__(self, mean=0., stddev=0.01):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape):
        return self.mean + np.random.randn(*shape) * self.stddev


class Zeros:
    def __call__(self, shape):
        return np.zeros(shape)


class Xavier:
    def __call__(self, shape):
        return np.random.randn(*shape) * np.sqrt(1 / shape[0])


class He:
    def __call__(self, shape):
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])
