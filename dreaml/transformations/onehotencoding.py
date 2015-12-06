from dreaml.dataframe.transform import BatchTransform
from dreaml.dataframe.dataframe import DataFrame
from scipy.sparse import lil_matrix,csr_matrix

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
