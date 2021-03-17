from layer_core import Layer
import numpy as np


class Identity(Layer):
    def forward(self, x):
        return x

    def backward(self, dout):
        return dout


class Relu(Layer):
    def __init__(self):
        self.__mask = None

    def forward(self, x):
        self.__mask = x <= 0
        out = x.copy()
        out[self.__mask] = 0
        return out

    def backward(self, dout):
        dout[self.__mask] = 0
        return dout


class Sigmoid(Layer):
    def __init__(self):
        self.__out = None

    def forward(self, x):
        self.__out = 1 / (1 + np.exp(-x))
        return self.__out

    def backward(self, dout):
        dx = dout * (1.0 - self.__out) * self.__out
        return dx


class Softmax(Layer):
    def __init__(self):
        self.__out = None

    def forward(self, x):
        x = x.T
        x = x - np.max(x, axis=0)
        e = np.exp(x)
        self.__out = e / np.sum(e, axis=0)
        self.__out = self.__out.T
        return self.__out

    def backward(self, dout):
        batch_size = self.__out.shape[0]
        input_size = self.__out.shape[1]
        dx = np.eye(input_size) * [np.diagflat(row) for row in self.__out] - [np.outer(row, row.T) for row in self.__out]
        dx = np.array([np.dot(dout[i], dx[i]) for i in range(batch_size)]) / batch_size
        return dx
