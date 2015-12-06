from dataframe.dataframe import DataFrame
import numpy as np
from random import randint

from abc import ABCMeta, abstractmethod
from exceptions import NotImplementedError


class Objective(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def f(target_df,*args,**kwargs):
        pass

    @abstractmethod
    def g(target_df,*args,**kwargs):
        pass

    def f_g(target_df,*args,**kwargs):
        return (f(target_df,*args,**kwargs),g(target_df,*args,**kwargs))

    @abstractmethod
    def structure(*args,**kwargs):
        pass