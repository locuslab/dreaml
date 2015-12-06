from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame
from pcabasis import PCABasis
from dot import Dot
_auto_dir = "auto/"
class PCA(BatchTransform):
    def func(self,target_df,X_df,num_bases=50):
        """ Project onto the PCA basis """
        # PCA function needs a df to store eigen basis in
        df = X_df._top_df
        pca_basis_location = (_auto_dir+"features/",_auto_dir+"pca_basis/")
        df[pca_basis_location] = PCABasis(X_df,num_bases)
        target_df.set_dataframe(Dot(X_df, df[pca_basis_location]))