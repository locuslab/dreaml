from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame
import numpy as np

class ZeroMean(BatchTransform):
    def func(self,target_df,X_df):
        X = X_df.get_matrix()
        X_m = np.mean(X, axis=0)
        X_zm = X - X_m # X with zero mean
        return DataFrame.from_matrix(X_zm)