from dreaml.dataframe.dataframe import DataFrame
import dreaml
import dreaml.loss as loss
import numpy as np
import numpy.random as nprand
from time import sleep
from dreaml.transformations import *
import numpy.linalg
import scipy.linalg

class TestTransformation:
    def setUp(self):
        self.item_count = 8
        assert self.item_count >= 4

    def test_dot(self):
        df = DataFrame()
        M1_path = ("row1/","col1/")
        M2_path = ("row2/","col2/")
        dot_path1 = ("row1/","col2/")
        M1 = nprand.rand(3,5)
        M2 = nprand.rand(5,8)
        df[M1_path] = DataFrame.from_matrix(M1)
        df[M2_path].set_matrix(M2)
        df[dot_path1] = Dot(df[M1_path],df[M2_path])
        assert (df[dot_path1].get_matrix()==M1.dot(M2)).all()

    def test_linear(self):
        df = DataFrame()
        M1_path = ("row1/","col1/")
        M2_path = ("row2/","col2/")
        linear_path1 = ("row1/","col2/")
        M1 = nprand.rand(3,5)
        M2 = nprand.rand(3,5)
        df[M1_path] = DataFrame.from_matrix(M1)
        df[M2_path].set_matrix(M2)
        a = 2
        b = -3
        df[linear_path1] = Linear(a,df[M1_path],b,df[M2_path])
        assert (df[linear_path1].get_matrix()==a*M1+b*M2).all()

    def test_permutation(self):
        df = DataFrame()
        M1_path = ("row1/","col1/")
        permute_path1 = ("row2/","col1/")
        M1 = nprand.rand(3,5)
        df[M1_path] = DataFrame.from_matrix(M1)
        df[permute_path1] = Permute(df[M1_path])
        p_df = df["auto/row1/","auto/permutation/"]
        p = p_df.get_matrix().ravel()
        assert (df[permute_path1].get_matrix()==M1[p,:]).all()

    def test_gd(self):
        df = DataFrame()
        M1_path = ("row1/","col1/")
        M2_path = ("row2/","col2/")
        batch1_path = ("row1/","col1/batch1/")
        batch2_path = ("row1/","col1/batch2/")
        x0_path = ("x0/","y0/")
        M1 = nprand.rand(3,5)
        df[batch1_path].set_matrix(M1)

        M2 = np.zeros((3,5))
        
        df[M2_path] = GD(loss.Toy,M2,None,y_df=df[M1_path])
        sleep(1)
        # df[M2_path].stop()
        assert np.allclose(df[M2_path].get_matrix(),df[M1_path].get_matrix())

        # Assert that the input structure has been replicated
        assert (df["row2/","col2/batch1/"].get_matrix()
                ==df[M2_path].get_matrix()).all()

        # Now attempt to extend the parameter matrix
        M3 = nprand.rand(3,4)
        df[batch2_path].set_matrix(M3)

        sleep(1)
        print df[M2_path].shape()
        print df[M1_path].shape()
        assert df[M2_path].shape()==df[M1_path].shape()
        assert df[M2_path].shape()==(3,9)
        assert np.allclose(df[M2_path].get_matrix(),df[M1_path].get_matrix())

        df[M2_path].stop()

    def test_sgd(self):
        # Also test sgd
        close = np.array([[-44.25076083,  38.62854577],
                          [-38.41473092,  36.29945225],
                          [-31.43300105,  30.79620632],
                          [-21.27706071,  24.08638079],
                          [-14.00259076,   6.54438641],
                          [ 11.52354442,  -6.07783327],
                          [ 48.69374796, -38.64696136],
                          [ 95.49682071, -84.38906967]])

        df = DataFrame()
        path = "row/","col/"
        df["xrow/","xcol/"]= DataFrame.from_matrix(np.arange(16).reshape(8,2))
        df["yrow/","ycol/"] = DataFrame.from_matrix(np.arange(8).reshape(8,1))
        X_df = df["xrow/","xcol/"]
        y_df = df["yrow/","ycol/"]
        df[path] = SGD(loss.Toy,close,X_df,y_df=y_df,batch_size=8,step_size=0.5)
        sleep(1)
        df[path].stop()
        assert np.allclose(df[path].get_matrix(), y_df.get_matrix())


    def test_zero_mean(self):
        df = DataFrame()
        M1_path = ("row1/","col1/")
        M2_path = ("row2/","col2/")
        M1 = nprand.rand(3,5)
        M1_zm = M1-np.mean(M1,axis=0)
        df[M1_path].set_matrix(M1)
        df[M2_path] = ZeroMean(df[M1_path])
        assert(df[M2_path].get_matrix()==M1_zm).all()

    def test_one_hot_encoding(self):
        df = DataFrame()
        M1_path = ("row1/","col1/")
        M2_path = ("row2/","col2/")
        n = 10
        m = 5
        M1 = np.vstack([nprand.randint(0,m,(n,1)),np.arange(m).reshape(m,1)])
        M2 = np.zeros((n+m,m))
        for i in range(n+m):
            M2[i,M1[i]] = 1

        df[M1_path].set_matrix(M1)
        df[M2_path] = OneHotEncoding(df[M1_path])

        assert(df[M2_path].get_matrix()==M2).all()

    def test_PCA_basis(self):
        df = DataFrame()
        M1_path = ("row1/","col1/")
        M2_path = ("row2/","col2/")
        n = 10
        m = 5
        d = 3
        M1 = nprand.rand(n,m)
        M1 = M1 - np.mean(M1,axis=0)
        # print M1
        df[M1_path].set_matrix(M1)

        df[M2_path] = PCABasis(df[M1_path],d)

        u,s,v_T = numpy.linalg.svd(M1,full_matrices=False)
        s[d+1:] = 0

        v = v_T.T[:,:d]

        M1_reconstructed = u.dot(np.diag(s).dot(v_T))
        # print M1
        # print M1_reconstructed

        M1_reconstructed2 = M1.dot(v).dot(v.T)
        # print M1_reconstructed2
        # print M1.dot(v.dot(v.T))


        covmat = (1./(n-1))*M1.T.dot(M1)
        evs, evmat = scipy.linalg.eig(covmat)
        p = np.argsort(evs)[::-1]
        evmat_sorted = evmat[:,p][:,:d]
        M1_reconstructed3 = M1.dot(evmat_sorted).dot(evmat_sorted.T)
        
        basis = df[M2_path].get_matrix()
        for i in range(evmat_sorted.shape[1]):
            assert np.isclose(basis[:,i], evmat_sorted[:,i]).all() or \
                   np.isclose(basis[:,i],-evmat_sorted[:,i]).all()

        M3_path = ("row3/","col3/")
        M3 = nprand.rand(2*n,m)
        M3 = M3 - np.mean(M1,axis=0)
        df[M3_path].set_matrix(M3)
        pca_path = ("pca/","pca/")
        df[pca_path] = PCA(df[M1_path],df[M3_path],d)
        pca = df[pca_path].get_matrix()
        proj = M3.dot(evmat_sorted)

        for i in range(pca.shape[1]):
            assert np.isclose(pca[:,i], proj[:,i]).all() or \
                   np.isclose(pca[:,i],-proj[:,i]).all()

