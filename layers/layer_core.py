import numpy as np
import abc
from abc import ABCMeta


class Layer(metaclass=ABCMeta):
    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self, dout):
        pass

    def is_optimizable(self):
        return False


class Affine(Layer):
    def __init__(self, shape, weight_initializer, bias_initializer):
        self.__x = None
        self.shape = shape
        self.W = weight_initializer(self.shape)
        self.b = bias_initializer((1, self.shape[1]))
        self.dW = None
        self.db = None

    def forward(self, x):
        self.__x = x
        out = np.dot(self.__x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.__x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

    def is_optimizable(self):
        return True
