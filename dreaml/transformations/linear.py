from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame

class Linear(BatchTransform):
    def func(self,target_df,a,X_df,b,Y_df,row_labels=None,col_labels=None):
        """Fetch matrices from dataframes, and return the resulting linear 
        combination in a dataframe"""
        x = X_df.get_matrix()
        y = Y_df.get_matrix()

        if row_labels==None:
            row_labels = X_df._row_index.keys()
        if col_labels==None:
            col_labels = X_df._col_index.keys()

        if (x.shape != y.shape):
            raise ValueError
        return DataFrame.from_matrix(a*x+b*y,row_labels,col_labels)