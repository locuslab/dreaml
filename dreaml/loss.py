from dataframe.dataframe import DataFrame
import numpy as np
from random import randint

# def _toy_loss(theta,y): 
#     return (np.array([[0.5*((theta-y)**2).sum()]]))

# def _toy_gradient(theta,y): 
#     return (theta-y)

# def _toy_loss(theta,y,fval=True,grad=True):
def toy(theta_df, X_df, y_df, fval=True,grad=True):
    if theta_df == None:
        return (y_df._row_index.keys(),y_df._col_index.keys())
    else:
        diff = theta_df.get_matrix()-y_df.get_matrix()
        f = (np.array([[0.5*(diff**2).sum()]]))
        g = diff
        return (f,g)

# # Purely for testing purposes...
# def toy(y_df):
#     def __toy_wrapper__(theta_df,i=None,batch_size=None):
#         return _toy_loss(theta_df.get_matrix(),y_df.get_matrix(readonly=True))
#     return __toy_wrapper__

def _multiclass_prob(theta,X):
    log_p = theta.dot(X.T)
    stable_log_p = log_p - log_p.max(axis=0)
    p = np.exp(stable_log_p)
    return np.divide(p,p.sum(axis=0))

# def _softmax_loss(theta,X,y,reg,fval=True,grad=True):
def softmax(theta_df,X_df,y_df,reg=0.01,fval=True,grad=True):
    if theta_df == None:
        rows = [str(k) for k in np.unique(y_df.get_matrix())]
        cols = X_df._col_index.keys()
        return rows,cols
    else:
        theta = theta_df.get_matrix()
        X = X_df.get_matrix(readonly=True)
        y = y_df.get_matrix(readonly=True)
        P = _multiclass_prob(theta,X)
        n = X.shape[0]
        if fval: 
            cost = np.mean(np.log([ P[y[i,0],i] for i in np.arange(n)]))
            penalty = reg/2*np.power(theta,2).sum()
            f = -cost+penalty
        else: 
            f = None
        if grad:
            P_minus = -P
            for k in range(len(y)):
                P_minus[y[k,0],k] += 1
            penalty = reg*theta
            total = np.zeros(theta.shape)
            for k in range(len(y)):
                total = total+np.kron(P_minus[:,k,None],X[k,:])
            g = -(total/len(y))+penalty
        else:
            g = None
        return (f,g)

# def softmax(X_df,y_df,reg):

#     def __softmax_wrapper__(theta_df,i=None,batch_size=None,fval=True,grad=True):
#         (n,m) = X_df.shape()
#         if i==None:
#             i = randint(0,n-1)
#         if batch_size==None:
#             batch_size=1
#         start = i
#         end = min(i+batch_size,n)
#         return _softmax_loss(theta_df.get_matrix(),
#                              X_df[start:end,:].get_matrix(readonly=True),
#                              y_df[start:end,:].get_matrix(readonly=True),
#                              reg,fval,grad)
#     return __softmax_wrapper__

# def _softmax_reg_loss(theta_df,X_df,y_df,reg):
#     theta = theta_df.get_matrix()
#     X = X_df.get_matrix()
#     y = y_df.get_matrix()
#     n = X.shape[0]
#     P = _multiclass_prob(theta,X)
#     cost = np.mean(np.log([ P[y[i],i] for i in np.arange(n)]))
#     penalty = reg/2*np.power(theta,2).sum()
#     return -cost+penalty


# def _softmax_reg_gradient(theta_df,X_df,y_df,reg):
#     theta = theta_df.get_matrix()
#     X = X_df.get_matrix()
#     y = y_df.get_matrix()
#     p = -_multiclass_prob(theta,X)
#     for k in range(len(y)):
#         p[y[k,0],k] += 1
#     penalty = reg*theta
#     total = np.zeros(theta.shape)
#     for k in range(len(y)):
#         total = total+np.kron(p[:,k,None],X[k,:])
#     return -(total/len(y))+penalty

# softmax_reg = (_softmax_reg_loss,_softmax_reg_gradient)