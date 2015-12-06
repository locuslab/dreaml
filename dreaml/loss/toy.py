from dreaml.objective import Objective
import numpy as np

class Toy(Objective):
    
    @staticmethod
    def f(theta_df,X_df,y_df=None):
        diff = theta_df.get_matrix()-y_df.get_matrix()
        return 0.5*(diff**2).sum()
    
    @staticmethod
    def g(theta_df,X_df,y_df=None):
        return theta_df.get_matrix()-y_df.get_matrix()
    
    @staticmethod
    def structure(X_df,y_df=None):
        return (y_df._row_index.keys(),y_df._col_index.keys())