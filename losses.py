import numpy as np
import abc
from abc import ABCMeta


class Loss(metaclass=ABCMeta):
    @abc.abstractmethod
    def forward(self, x, y):
        pass

    @abc.abstractmethod
    def backward(self):
        pass


class CrossEntropy(Loss):
    def __init__(self):
        self.__x = None
        self.__y = None

    def forward(self, x, y):
        if x.size == y.size:
            y = y.argmax(axis=1)
        self.__y = y.astype(int).flatten()
        self.__x = x

        batch_size = self.__y.size
        return -np.sum(np.log(x[np.arange(batch_size), self.__y] + 1e-7)) / batch_size

    def backward(self):
        y = np.zeros(self.__x.shape)
        y[np.arange(self.__y.size), self.__y] = 1.
        return -y / self.__x


# Binary Classification
# Sigmoid Activation + CrossEntropy Loss
class SigmoidLoss(Loss):
    def __init__(self):
        self.__out = None
        self.__y = None

    def forward(self, x, y):
        self.__y = y.reshape(*x.shape).astype(int)

        e = np.exp(-x)
        self.__out = 1 / (1 + e)

        batch_size = self.__y.size
        return -np.sum(-x + x * self.__y - np.log(1 + e) + 1e-7) / batch_size

    def backward(self):
        batch_size = self.__y.size
        dx = -self.__y / self.__out + (1 - self.__y) / (1 - self.__out)
        dx /= batch_size
        return dx


# Multi-class Classification
# Softmax Activation + CrossEntropy Loss
class SoftmaxLoss(Loss):
    def __init__(self):
        self.__out = None
        self.__y = None

    def forward(self, x, y):
        if y.size == x.size:
            y = y.argmax(axis=1)
        self.__y = y.astype(int).flatten()

        x = x.T
        x = x - np.max(x, axis=0)
        e = np.exp(x)
        self.__out = e / np.sum(e, axis=0)
        self.__out = self.__out.T

        batch_size = self.__y.size
        return -np.sum(np.log(self.__out[np.arange(batch_size), self.__y] + 1e-7)) / batch_size

    def backward(self):
        batch_size = self.__y.size
        dx = self.__out.copy()
        dx[np.arange(batch_size), self.__y] -= 1
        dx = dx / batch_size
        return dx
