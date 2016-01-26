from dreaml.dataframe.transform import ContinuousTransform
from numpy.random import randint
import numpy as np
from scipy.sparse import csr_matrix

class kmpp(ContinuousTransform):
    """Run K-means plus plus."""
    __module__ = "dreaml.transformations"

    def init_func(self,target_df,X_df,k):
        self.centers = self.initial_centers(X_df.get_matrix(),k)

    def continuous_func(self,target_df,X_df,k):
        """ Run standard K-means """
        X = X_df.get_matrix()
        n = X.shape[0]
        labels = closest_centers(self,X)
        ohe = csr_matrix(np.ones(n),(np.arange(n),labels),(n,k))
        self.centers = ohe.T*X/(ohe.sum(axis=0).reshape(k,1))


    def closest_centers(self,X):
        n = X.shape[0]
        center_norms = (self.centers*self.centers).sum(axis=X.ndim-1)
        return np.apply_along_axis(min_sq_dist,
                                   X.ndim-1,
                                   X,
                                   self.centers,
                                   center_norms)

    def min_sq_dist(x,centers,center_norms):
        return np.argmin(-2*x.dot(centers.T)+center_norms)

    def initial_centers(self,X,k):
        """ Run K-means plus plus initialization """
        # Currently random
        return randint(k,X.shape[0])

