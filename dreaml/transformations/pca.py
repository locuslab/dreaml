from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame
from pcabasis import PCABasis
from dot import Dot
import numpy as np
import numpy.linalg as la
_auto_dir = "auto/"
class PCA(BatchTransform):
    def func(self,target_df,X_pca_df,X_full_df=None,num_bases=50):
        """ Project onto the PCA basis """
        if X_full_df == None:
            X_full_df = X_pca_df
            
        X_mean = np.mean(X_pca_df.r_matrix,axis=0)

        # the PCA basis is exposed for the user
        self.v = self.pca_basis(X_pca_df.r_matrix,num_bases)

        # Use numbers as the label for the basis dimension
        col_labels = [str(i) for i in range(num_bases)]

        # Set the matrix with the corresponding labels
        target_df.set_matrix((X_full_df.r_matrix-X_mean).dot(self.v),
                             row_labels=X_full_df.rows(),
                             col_labels=col_labels)

    def pca_basis(self,X, num_bases=50):
        X_m = np.mean(X,axis=0) # mean
        X_zm = X - X_m # X with 0 mean
        u,s,v_T = la.svd(X_zm)
        return np.real(v_T.T[:,:num_bases])