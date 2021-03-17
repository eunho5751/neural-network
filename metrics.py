import abc
from abc import ABCMeta
import numpy as np


class Metrics(metaclass=ABCMeta):
    @abc.abstractmethod
    def result(self, y_true, y_pred):
        pass


class CategoricalAccuracy(Metrics):
    def result(self, y_true, y_pred):
        true_indices = np.argmax(y_true, axis=1) if y_true.shape[1] > 1 else y_true.flatten()
        pred_indices = np.argmax(y_pred, axis=1)
        acc = np.sum(true_indices == pred_indices) / y_true.shape[0]
        return acc


class BinaryAccuracy(Metrics):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def result(self, y_true, y_pred):
        acc = np.sum(y_true == (y_pred >= self.threshold)) / y_true.shape[0]
        return acc
