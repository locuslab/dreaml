from dreaml.dataframe.transform import ContinuousTransform
from time import sleep
import numpy as np

class Metrics(ContinuousTransform):
    """ Computes the given metrics
    """
    def __init__(self,metrics_list,*args,**kwargs):
        self.metrics_names = kwargs.pop('metrics_names',\
                                    [str(i) for i in range(len (metrics_list))])
        if len(self.metrics_names) != len(metrics_list):
            raise ValueError("List of names must have same dimension as list\
                of metrics")

        self.interval = kwargs.pop('interval',1)

        super(Metrics,self).__init__(metrics_list,
                                     *args,
                                     **kwargs)

    def init_func(self,target_df,metrics_list,*args,**kwargs):
        target_df.set_structure(target_df.rows(),self.metrics_names)
        target_df.set_matrix(np.zeros((target_df.shape()[0],len(metrics_list))))

    def continuous_func(self,target_df,metrics_list,*args,**kwargs):
        metrics = target_df.get_matrix()
        for i,metric in enumerate(metrics_list):
            # name = self.metrics_names[i]
            m = metrics_list[i](*args,**kwargs)
            metrics[:,i] = m

        sleep(self.interval)