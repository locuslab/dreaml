from dataframe.dataframe import DataFrame
from dataframe.transform import Transform,ContinuousTransform,BatchTransform

from scipy.sparse import lil_matrix,csr_matrix

import numpy as np
import numpy.linalg as la
import numpy.random as random
from time import time,sleep

import matplotlib.pyplot as plt

_auto_dir = "auto/"

class Identity(BatchTransform):
    def func(self,target_df, X_df):
        return DataFrame.from_matrix(X_df.get_matrix())

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

class ZeroMean(BatchTransform):
    def func(self,target_df,X_df):
        X = X_df.get_matrix()
        X_m = np.mean(X, axis=0)
        X_zm = X - X_m # X with zero mean
        return DataFrame.from_matrix(X_zm)

# TODO: permute should not merge partitions that should be of different types!
# I.e. integer labels and float features

class Permute(BatchTransform):
    def func(self,target_df,X_df):
        X = X_df.get_matrix()
        P = np.random.permutation(X.shape[0])
        row_labels = X_df._row_index.keys()
        col_labels = X_df._col_index.keys()

        (row_query,col_query) = X_df.pwd()

        X_df._top_df[_auto_dir+row_query,_auto_dir+"permutation/"] = \
            DataFrame.from_matrix(P[:,None])
        # print "finished permute"
        return DataFrame.from_matrix(X[P,:],row_labels,col_labels)

class OneHotEncoding(BatchTransform):
    def func(self,target_df,Y_df):
        print "OHE....."
        Y = Y_df.get_matrix().flatten().tolist()
        d = {}
        for y in Y:
            if y not in d:
                d[y] = len(d)
        for i,y in enumerate(sorted(d.keys())):
            print i,y
            d[y] = i
        OHE = lil_matrix((len(Y),len(d)))
        for i,y in enumerate(Y):
            OHE[i,d[y]] = 1
        
        row_labels =Y_df._row_index.keys()
        col_labels = [str(i) for i in range(len(d))]

        return DataFrame.from_matrix(OHE.tocsr(),row_labels,col_labels)

class KitchenSinks(BatchTransform):
    def __init__(self,X_df,n_rbf):
        super(KitchenSinks,self).__init__(X_df,n_rbf)

        X = X_df.get_matrix()
        n_trials = int(X.shape[0]**1.5)
        I = random.randint(0, X.shape[0], n_trials)
        deltI = random.randint(1, X.shape[0], n_trials)
        J = (I + deltI) % X.shape[0]
        dists = sorted(map(lambda i : la.norm(X[I[i],:] - X[J[i],:]), range(n_trials)))
        s = dists[n_trials / 2]
        
        self.s = float(s)

    def func(self,target_df,X_df,n_rbf):
        X = X_df.get_matrix()
        s = self.s

        W = random.randn(X.shape[1], n_rbf) / s / np.sqrt(2)
        B = random.uniform(0, 2*np.pi, n_rbf)

        return DataFrame.from_matrix(np.cos(X.dot(W)+ B))

class GD(ContinuousTransform):
    def __init__(self,*args,**kwargs):
        print args,kwargs
        super(GD,self).__init__(*args,**kwargs)
        self.niters = 0

    def continuous_func(self,target_df,Obj,x0,*args,**kwargs):
        self.niters += 1
        res = Obj.g(target_df,*args,**kwargs)

        P = target_df.get_matrix()

        P -= 0.1*res

    def init_func(self,target_df,Obj,x0,*args,**kwargs):
        print args
        print kwargs
        rows,cols = Obj.structure(*args,**kwargs)
        target_df.set_structure(rows,cols)
        if x0.shape == target_df.shape():
            target_df.set_matrix(x0)

class SGD(ContinuousTransform):
    """ Returns a Transformation that runs SGD for a given loss ``f`` at initial
    point ``x0``. By default, it will run with a minibatch size of 1, unless
    keyword argument ``batch_size`` is provided. 

    The function ``f`` should follow the specification of a loss function. 

    Special keyword arguments:
    ==========  =======  
    kwarg       default    
    ==========  =======  
    step_size   1e-4
    batch_size  1 
    ==========  =======  

    """
    def __init__(self,Obj,x0,*args,**kwargs):
        self.batch_size = kwargs.pop('batch_size',1)
        self.step_size = kwargs.pop('step_size',1e-4)

        super(SGD,self).__init__(Obj,x0,*args,**kwargs)
        self.niters = 0
        self.batch = 0

    def init_func(self, target_df,Obj,x0,*args,**kwargs):

        if len(args)==0:
            raise ValueError("Mini-batchable arguments must be provided. If\
                none are necessary, consider using the gradient descent (GD)\
                transformation instead.")
        print args,kwargs
        rows,cols = Obj.structure(*args,**kwargs)
        target_df.set_structure(rows,cols)
        if x0.shape == target_df.shape():
            target_df.set_matrix(x0)

    def continuous_func(self, target_df,Obj,x0,*args,**kwargs):
        n = args[0].shape()[0]

        start = self.batch
        end = min(start+self.batch_size,n)
        g = Obj.g(target_df,*[df[start:end,:] if isinstance(df,DataFrame)
            else df for df in args],**kwargs)

        Theta = target_df.get_matrix()

        self.niters +=1
        self.batch += self.batch_size
        if self.batch >= n:
            self.batch = 0
        Theta -= self.step_size*g

class PCABasis(BatchTransform):
    def func(self,target_df,X_df, num_bases=50):
        X = X_df.get_matrix()
        X_m = np.mean(X,axis=0) # mean
        X_zm = X - X_m # X with 0 mean
        u,s,v_T = la.svd(X_zm)

        row_labels = [str(i) for i in range(X.shape[1])]
        col_labels = [str(i) for i in range(num_bases)]
        return DataFrame.from_matrix(np.real(v_T.T[:,:num_bases]),row_labels,col_labels)

class PCA(BatchTransform):
    def func(self,target_df,X_df,num_bases=50):
        """ Project onto the PCA basis """
        # PCA function needs a df to store eigen basis in
        df = X_df._top_df
        pca_basis_location = (_auto_dir+"features/",_auto_dir+"pca_basis/")
        df[pca_basis_location] = PCABasis(X_df,num_bases)
        target_df.set_dataframe(Dot(X_df, df[pca_basis_location]))

def __softmax_error_function__(target_df,params_df,X_df,y_df):
    n = X_df.shape()[0]
    Theta = params_df.get_matrix()
    X = X_df.get_matrix()
    y = y_df.get_matrix()
    P = Theta.dot(X.transpose())
    P = np.exp(P)
    P = np.divide(P, P.sum(axis=0))
    y_p = np.argmax(P,axis=0)
    return DataFrame.from_matrix(np.array(float(np.sum(np.not_equal(np.squeeze(y), y_p)))/n).reshape(1,1))

def softmax_error(params_df,X_df,y_df,subroutine=False):
    return Transform(__softmax_error_function__,None,False,subroutine,
                     params_df,X_df,y_df)
