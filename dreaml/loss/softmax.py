from dreaml.objective import Objective
import numpy as np

class Softmax(Objective):
    
    @staticmethod
    def f(theta_df,X_df,y_df,reg=0.01):
        theta = theta_df.get_matrix()
        penalty = reg/2*np.power(theta,2).sum()
        # X = X_df.get_matrix(readonly=True)
        # y = y_df.get_matrix(readonly=True)
        # P = Softmax._multiclass_prob(theta,X)
        # n = X.shape[0]

        # cost = np.mean(np.log([ P[y[i,0],i] for i in np.arange(n)]))
        cost = np.mean(Softmax.f_vec(theta_df,X_df,y_df,reg=reg))
        return -cost+penalty
    
    @staticmethod
    def f_vec(theta_df,X_df,y_df,reg=0.01):
        theta = theta_df.get_matrix()
        X = X_df.get_matrix(readonly=True)
        y = y_df.get_matrix(readonly=True)
        P = Softmax._multiclass_prob(theta,X)
        n = X.shape[0]

        return np.log([ P[y[i,0],i] for i in np.arange(n)])

    @staticmethod
    def g(theta_df,X_df,y_df,reg=0.01):
        theta = theta_df.get_matrix()
        X = X_df.get_matrix(readonly=True)
        y = y_df.get_matrix(readonly=True)
        P = Softmax._multiclass_prob(theta,X)
        n = X.shape[0]

        P_minus = -P
        for k in range(len(y)):
            P_minus[y[k,0],k] += 1
        penalty = reg*theta
        total = np.zeros(theta.shape)
        for k in range(len(y)):
            total = total+np.kron(P_minus[:,k,None],X[k,:])
        return -(total/len(y))+penalty
    
    @staticmethod
    def structure(X_df,y_df,reg=0.01):
        rows = [str(k) for k in np.unique(y_df.get_matrix())]
        cols = X_df._col_index.keys()
        return rows,cols

    @staticmethod
    def err(theta_df,X_df,y_df,reg=0.01):
        theta = theta_df.get_matrix()
        X = X_df.get_matrix(readonly=True)
        y = y_df.get_matrix(readonly=True)
        n = X.shape[0]
        P = Softmax._multiclass_prob(theta,X)
        y_pred = np.argmax(P,axis=0)
        return np.equal(np.squeeze(y),y_pred)

    @staticmethod
    def _multiclass_prob(theta,X):
        log_p = theta.dot(X.T)
        stable_log_p = log_p - log_p.max(axis=0)
        p = np.exp(stable_log_p)
        return np.divide(p,p.sum(axis=0))
