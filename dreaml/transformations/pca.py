from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame
from pcabasis import PCABasis
from dot import Dot
import numpy as np
_auto_dir = "auto/"
class PCA(BatchTransform):
    def func(self,target_df,X_pca_df,X_full_df,num_bases=50):
        """ Project onto the PCA basis """
        # PCA function needs a df to store eigen basis in
        df = target_df._top_df
        X_pca = X_pca_df.get_matrix()
        X_mean = np.mean(X_pca,axis=0)
        pca_basis_location = (_auto_dir+"features/",_auto_dir+"pca_basis/")
        df[pca_basis_location] = PCABasis(X_pca_df,num_bases)
        v = df[pca_basis_location].get_matrix()
        X_full = X_full_df.get_matrix()
        output = DataFrame.from_matrix((X_full-X_mean).dot(v),
                                       row_labels=X_full_df.rows(),
                                       col_labels=df[pca_basis_location].cols())
        target_df.set_dataframe(output)
