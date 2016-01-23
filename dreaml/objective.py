from dataframe.dataframe import DataFrame
import numpy as np
from random import randint

from abc import ABCMeta, abstractmethod

from loss import Softmax
from loss import Square


class Objective(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, target_df, *args, **kwargs):
        pass
    
    @abstractmethod
    def f(self):
        pass

    @abstractmethod
    def g(self):
        pass

    @abstractmethod
    def structure(*args,**kwargs):
        pass

class SquareTest(Objective):
    def __init__(self, y_prediction_df, y_target_df):
        self.loss = Square(y_prediction_df.get_matrix(),y_target_df.get_matrix()) 
        self.n = y_target_df.shape[0]
        self.yt_df = y_target_df

    def f(self): return np.mean(self.loss.f())
    def g(self): return self.loss.g()/self.n
    def h(self): pass

    @staticmethod
    def structure(y_target_df): return y_target_df.rows(),y_target_df.cols()

class SoftmaxRegression(Objective):
    def __init__(self,theta_df,X_df,y_df,reg):
        """ Combine Softmax loss with a linear model for Softmax regression. """
        self.X = X_df.get_matrix()
        self.theta = theta_df.get_matrix()

        yt = y_df.get_matrix()
        yp = self.X.dot(self.theta.T)
        self.loss = Softmax(yp,yt)
        
        self.X_df = X_df
        self.y_df = y_df
        self.reg = reg

    def f(self): 
        return np.mean(self.loss.f()) + self.reg/2*np.sum(self.theta**2)
    def f_vec(self):
        return self.loss.f() + self.reg/2*np.sum(self.theta**2)
    def g(self): 
        n = self.X.shape[0]
        return self.loss.g().T.dot(self.X)/n + self.reg*self.theta
    def h(self): 
        # Needs to be averaged and also regularized
        return self.loss.h()

    @staticmethod
    def structure(X_df,y_df,reg):
        return y_df.cols(), X_df.rows()