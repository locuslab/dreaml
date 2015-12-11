from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame


class Dot(BatchTransform):
    """ Calculates the dot product of the contents of X_df and y_df """
    __module__ = "dreaml.transformations"

    def __init__(self,X_df,y_df):
        super(Dot,self).__init__(X_df,y_df)

    def func(self,target_df,X_df,Y_df):
        x = X_df.get_matrix()
        y = Y_df.get_matrix()
        row_labels = X_df._row_index.keys()
        col_labels = Y_df._col_index.keys()
        return DataFrame.from_matrix(x.dot(y),row_labels,col_labels)