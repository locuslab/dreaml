from dreaml.dataframe.transform import ContinuousTransform
from numpy.random import randint
import numpy as np
from scipy.sparse import csr_matrix
from numpy.random import choice

class KMPP(ContinuousTransform):
    """Run K-means plus plus."""
    __module__ = "dreaml.transformations"

    def init_func(self,target_df,X_df,k):
        M = KMPP.initial_centers(X_df.get_matrix(),k)
        print M
        target_df.set_matrix(M)

    def continuous_func(self,target_df,X_df,k):
        """ Run standard K-means """
        X = X_df.get_matrix()
        n = X.shape[0]
        labels = KMPP.closest_centers(X,target_df.get_matrix())
        ohe = csr_matrix((np.ones(n),(np.arange(n),labels)),shape=(n,k))
        target_df.set_matrix(ohe.T*X/(np.asarray(ohe.sum(axis=0)).reshape(k,1)))

    @staticmethod
    def closest_centers(X,centers):
        n = X.shape[0]
        center_norms = (centers*centers).sum(axis=X.ndim-1)
        return np.apply_along_axis(KMPP.argmin_sq_dist,
                                   X.ndim-1,
                                   X,
                                   centers,
                                   center_norms)

    @staticmethod
    def argmin_sq_dist(x,centers,center_norms):
        return np.argmin(-2*x.dot(centers.T)+center_norms)

    @staticmethod
    def initial_centers(X,k):
        """ Run K-means plus plus initialization """
        n = X.shape[0]
        j = randint(n)
        centers = np.zeros((k,X.shape[X.ndim-1]))
        centers[0,:] = X[j,:]
        indices = [j]
        for i in range(1,k):
            center_norms = (centers*centers).sum(axis=X.ndim-1)
            d = np.apply_along_axis(KMPP.min_sq_dist,
                                    X.ndim-1,
                                    X,
                                    centers[:i],
                                    center_norms[:i])
            d[indices] = 0
            j = choice(n,p=d/d.sum())
            centers[i,:] = X[j,:]
            indices.append(j)
        return centers


    @staticmethod
    def min_sq_dist(x,centers,center_norms):
        return np.min((x*x).sum()-2*x.dot(centers.T)+center_norms)
