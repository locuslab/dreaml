from dataframe.dataframe import DataFrame
import numpy as np
from random import randint

from abc import ABCMeta, abstractmethod
from exceptions import NotImplementedError


class Loss(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, y_prediction, y_target):
        pass

    @abstractmethod
    def f(self):
        pass

    @abstractmethod
    def g(self):
        pass

    @abstractmethod
    def h(self):
        pass


class Logistic(Loss):
    """ The logistic loss function:
    loss(y_p,y_t) = log(1 + exp(- y_p * y_t))
    where y_p is real-valued number of y_t in {-1, +1}
    """
    def __init__(self, y_prediction, y_target):
        self.z = 1/(1+np.exp(y_prediction * y_target))
        self.yt = y_target
    def f(self): return np.log(1-self.z)
    def g(self): return -self.yt/self.z
    def h(self): return self.z*(1-self.z)

class Hinge(Loss):
    """ The hinge loss:
    loss(y_p, y_t) = max(0, 1-y_p*y_t)
    where y_p is real-valued and y_t in {-1, +1}
    """
    def __init__(self, y_predidction, y_target):
        self.z = 1-y_prediction*y_target
        self.yt = y_target
    def f(self): return np.maximum(0,self.z)
    def g(self): return -self.yt*float(self.z > 0)
    def h(self): return 0


def Square(Loss):
    def __init__(self, y_predcition, y_target):
        self.r = y_prediction - y_target
    def f(self): return 0.5*self.r**2
    def g(self): return self.r
    def h(self): return 1.


class Softmax(Loss):
    """ The softmax loss:
    loss(y_p, y_t) = logsumexp(y_p) - y_t*y_p
    where y_p is a vector of real-valued numbers and y_t is a binary indicator.
    """
    def __init__(self, y_prediction, y_target):
        self.z = np.exp(y_prediction)
        self.sz = np.sum(self.z,axis=y_prediction.ndim-1)
        self.h = None
        self.yt = y_target
    def f(self): return np.log(self.sz)
    def g(self): 
        if self.h is None:
            self.h = self.z / self.sz[:,None]
        return self.h - self.yt
    def h(self):
        if self.h is None:
            self.h = self.z / self.sz[:,None]
        return np.array([np.diag(h) - np.outer(h,h) for h in self.h])










