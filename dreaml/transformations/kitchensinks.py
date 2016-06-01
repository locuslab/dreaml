from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame
import numpy as np
import numpy.random as random
import numpy.linalg as la

_auto_dir = "auto/"

class KitchenSinks(BatchTransform):
    def func(self,target_df,X_df,num_features=1):
        X = X_df.get_matrix(readonly=True)
        n_trials = int(X.shape[0]**1.5)
        I = random.randint(0, X.shape[0], n_trials)
        deltI = random.randint(1, X.shape[0], n_trials)
        J = (I + deltI) % X.shape[0]
        dists = sorted(map(lambda i : la.norm(X[I[i],:] - X[J[i],:]), range(n_trials)))
        s = dists[n_trials / 2]

        self.s = float(s)

        W = random.randn(X.shape[1], num_features) / s / np.sqrt(2)
        B = random.uniform(0, 2*np.pi, num_features)

        target_df.set_matrix(np.cos(X.dot(W)+ B),
                             row_labels=X_df.rows())