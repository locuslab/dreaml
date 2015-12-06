from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame
import numpy as np

_auto_dir = "auto/"

class Permute(BatchTransform):
    def func(self,target_df,X_df):
        X = X_df.get_matrix()
        P = np.random.permutation(X.shape[0])
        row_labels = X_df._row_index.keys()
        col_labels = X_df._col_index.keys()

        (row_query,col_query) = X_df.pwd()

        X_df._top_df[_auto_dir+row_query,_auto_dir+"permutation/"] = \
            DataFrame.from_matrix(P[:,None])
        # print "finished permute"
        return DataFrame.from_matrix(X[P,:],row_labels,col_labels)
