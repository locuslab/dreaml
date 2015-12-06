from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame

class Dot(BatchTransform):
    def func(self,target_df,X_df,Y_df):
        """Fetch matrices from dataframes, and return the resulting dot product
        in a dataframe
        """
        x = X_df.get_matrix()
        y = Y_df.get_matrix()
        row_labels = X_df._row_index.keys()
        col_labels = Y_df._col_index.keys()
        return DataFrame.from_matrix(x.dot(y),row_labels,col_labels)