from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame
# from scipy.sparse import lil_matrix,csr_matrix
import numpy as np

class OneHotEncoding(BatchTransform):
    def func(self,target_df,Y_df):
        Y = Y_df.get_matrix().squeeze().tolist()
        d = {}
        for y in Y:
            if y not in d:
                d[y] = len(d)
        for i,y in enumerate(sorted(d.keys())):
            d[y] = i
        OHE = np.zeros((len(Y),len(d)),dtype=bool)
        for i,y in enumerate(Y):
            OHE[i,d[y]] = True
        
        row_labels =Y_df._row_index.keys()
        col_labels = [str(i) for i in range(len(d))]

        print OHE.shape
        target_df.set_matrix(OHE,row_labels,col_labels)
