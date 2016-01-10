from dreaml.dataframe.dataframe import DataFrame
import dreaml
from dreaml.loss import *
from dreaml.objective import SoftmaxRegression
import numpy as np
import numpy.random as nprand
from dreaml.loss.softmax import Softmax

class TestLoss:
    def setUp(self):
        self.item_count = 8
        assert self.item_count >= 4

    def central_diff(self,f,epsilon,theta):
        print epsilon
        x = theta.get_matrix()
        n = x.shape[0]
        g = np.zeros(x.shape)
        if x.ndim == 2:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    upper = x.copy()
                    upper[i,j] += epsilon
                    lower = x.copy()
                    lower[i,j] -= epsilon
                    g[i,j] = ((f(DataFrame.from_matrix(upper))
                             -f(DataFrame.from_matrix(lower)))
                             /(2*epsilon))
        elif x.ndim == 1:
            for i in range(x.shape[0]):
                upper = x.copy()
                upper[i] += epsilon
                lower = x.copy()
                lower[i] -= epsilon

                g[i] = ((f(DataFrame.from_matrix(upper))
                       -f(DataFrame.from_matrix(lower)))
                       /(2*epsilon))
        else:
            raise ValueError
        return g


    # def test_toy_loss(self):
    #     df = DataFrame()
    #     epsilon = 1e-4
    #     y_path = ("y/","y/")
    #     theta_path = ("theta/","theta/")
    #     y = nprand.rand(10,1)
    #     theta = nprand.rand(10,1)
    #     df[y_path] = DataFrame.from_matrix(y)
    #     df[theta_path] = DataFrame.from_matrix(theta)
    #     # df[X_path] = DataFrame.from_matrix(nprand.rand(10,2))

    #     toy = lambda theta_df: Toy.f(theta_df,None,y_df=df[y_path])

    #     g0 = self.central_diff(toy,epsilon,df[theta_path])
    #     g = Toy.g(df[theta_path],None,y_df=df[y_path])

    #     assert(np.allclose(g,g0))

    def test_softmax_reg_loss(self):
        df = DataFrame()
        epsilon = 1e-4
        y_path = ("y/","y/")
        theta_path = ("theta/","theta/")
        X_path = ("X/","X/")

        k = 10
        n,m = 5,8
        df[X_path] = DataFrame.from_matrix(nprand.rand(n,m))
        df[theta_path] = DataFrame.from_matrix(nprand.rand(k,m))
        y = np.zeros((n,k),dtype=bool)
        for i in range(n):
            j = nprand.randint(k)
            y[i,j] = True
        df[y_path] = DataFrame.from_matrix(y)
        reg = 0.0001

        softmax = lambda theta_df: SoftmaxRegression(theta_df, df[X_path], 
                                                df[y_path], reg).f()


        g_central = self.central_diff(softmax,epsilon,df[theta_path])
        g1 = SoftmaxRegression(df[theta_path], df[X_path], df[y_path], reg).g()

        # print g_central
        assert(np.allclose(g_central,g1))

        # Test batch by checking average gradient
        # g2 = np.zeros((k,m))
        # for i in range(n):
        #     g2 += Softmax.g(df[theta_path], df[X_path], df[y_path], reg)
        # g2 /= n
        # assert(np.allclose(g_central,g2))