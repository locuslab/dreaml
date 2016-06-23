from dreaml.dataframe.transform import ContinuousTransform
from dreaml.dataframe.dataframe import DataFrame

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
        rows,cols = Obj.structure(*args,**kwargs)
        if target_df.empty():
            target_df.set_structure(rows,cols)
            if x0.shape == target_df.shape:
                target_df.set_matrix(x0)
        else: 
            target_df.set_structure(rows,cols)

        # reinsert into cache
        for df in args:
            if isinstance(df,DataFrame):
                df.get_matrix()

    def continuous_func(self, target_df,Obj,x0,*args,**kwargs):
        n = args[0].shape[0]

        start = self.batch
        end = min(start+self.batch_size,n)
        g = Obj(target_df,*[df[start:end,:] if isinstance(df,DataFrame)
            else df for df in args],**kwargs).g()

        self.niters +=1
        self.batch += self.batch_size
        if self.batch >= n:
            self.batch = 0
        target_df.rw_matrix -= self.step_size*g