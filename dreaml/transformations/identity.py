from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame

class Identity(BatchTransform):
    def func(self,target_df, X_df):
        return DataFrame.from_matrix(X_df.get_matrix())