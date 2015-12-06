from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame

import numpy as np
import numpy.linalg as la

class PCABasis(BatchTransform):
    def func(self,target_df,X_df, num_bases=50):
        X = X_df.get_matrix()
        X_m = np.mean(X,axis=0) # mean
        X_zm = X - X_m # X with 0 mean
        u,s,v_T = la.svd(X_zm)

        row_labels = [str(i) for i in range(X.shape[1])]
        col_labels = [str(i) for i in range(num_bases)]
        return DataFrame.from_matrix(np.real(v_T.T[:,:num_bases]),row_labels,col_labels)

