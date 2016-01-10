from dataframe.dataframe import DataFrame
import numpy as np
from random import randint

from abc import ABCMeta, abstractmethod

from loss import Softmax


class Objective(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def f(target_df,*args,**kwargs):
        pass

    @abstractmethod
    def g(target_df,*args,**kwargs):
        pass

    @abstractmethod
    def structure(*args,**kwargs):
        pass

class SoftmaxRegression(Objective):
    def __init__(self,theta_df,X_df,y_df,reg):
        X = X_df.get_matrix()
        theta = theta_df.get_matrix()
        yt = y_df.get_matrix()
        # yp = theta.dot(X.T)
        yp = X.dot(theta.T)

        self.loss = Softmax(yp,yt)
        self.theta = theta
        self.X = X
        self.X_df = X_df
        self.y_df = y_df
        self.reg = reg

    def f(self): 
        return np.mean(self.loss.f()) + self.reg/2*np.sum(self.theta**2)
    def g(self): 
        n = self.X.shape[0]
        return self.loss.g().T.dot(self.X)/n + self.reg*self.theta
    def h(self): 
        # Needs to be averaged and also regularized
        return self.loss.h()
    def structure(self):
        return self.y_df.cols(), self.X_df.rows()