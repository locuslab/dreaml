from dreaml.dataframe.transform import ContinuousTransform

class GD(ContinuousTransform):
    def __init__(self,*args,**kwargs):
        super(GD,self).__init__(*args,**kwargs)
        self.niters = 0

    def continuous_func(self,target_df,Obj,x0,*args,**kwargs):
        self.niters += 1
        res = Obj.g(target_df,*args,**kwargs)

        P = target_df.get_matrix()

        P -= 0.1*res

    def init_func(self,target_df,Obj,x0,*args,**kwargs):
        rows,cols = Obj.structure(*args,**kwargs)
        target_df.set_structure(rows,cols)
        if x0.shape == target_df.shape():
            target_df.set_matrix(x0)
