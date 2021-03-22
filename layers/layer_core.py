import numpy as np
import abc
from abc import ABCMeta
import initializers
import utils


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
    def __init__(self, shape, weight_initializer=initializers.RandomNormal(), bias_initializer=initializers.Zeros()):
        self.__x = None
        self.shape = shape
        self.original_shape = None
        self.W = weight_initializer(self.shape)
        self.b = bias_initializer((1, self.shape[1]))
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_shape = x.shape
        self.__x = x.reshape(x.shape[0], -1)
        out = np.dot(self.__x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        dx = dx.reshape(*self.original_shape)
        self.dW = np.dot(self.__x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

    def is_optimizable(self):
        return True


class Convolution(Layer):
    def __init__(self, filter_num, channel_num, kernel_size, strides=1, padding=0, kernel_initializer=initializers.RandomNormal(), bias_initializer=initializers.Zeros()):
        self.filter_num = filter_num
        self.channel_num = channel_num
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.W = kernel_initializer((filter_num, channel_num, kernel_size, kernel_size))
        self.b = bias_initializer((filter_num,))
        self.dW = None
        self.db = None

        self.x = None
        self.col = None
        self.col_W = None

    def forward(self, x):
        N, C, H, W = x.shape
        FN, C, KH, KW = self.W.shape

        out_h = (H + 2*self.padding - KH) // self.strides + 1
        out_w = (W + 2*self.padding - KW) // self.strides + 1

        col = utils.im2col(x, (KH, KW), self.strides, self.padding)  # (N * out_h * out_w, C * kernel_size * kernel_size)
        col_W = self.W.reshape(FN, -1).T  # (C * KH * KW, FN)

        out = np.dot(col, col_W) + self.b  # (N * out_h * out_w, FN)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)  # (N, FN, out_h, out_w)

        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        FN, C, KH, KW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)  # (N, out_h, out_w, FN) -> (N * out_h * outW, FN)

        self.dW = np.dot(self.col.T, dout)  # (C * kernel_size * kernel_size, FN)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, KH, KW)  # (FN, C * kernel_size * kernel_size) -> (FN, C, KH, KW)
        self.db = np.sum(dout, axis=0)

        dcol = np.dot(dout, self.col_W.T)
        dx = utils.col2im(dcol, self.x.shape, (KH, KW), self.strides, self.padding)
        return dx

    def is_optimizable(self):
        return True
