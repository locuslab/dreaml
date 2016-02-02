from abc import ABCMeta, abstractmethod
from threading import Thread
from time import sleep, time

class Transform(object): 
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def apply(self,target_df=None):
        """ Apply func(df, ...) and return the resulting dataframe

        Some restrictions: 
            1. func must take in df as the first argument (this is to
                allow for automated generation of entries when necessary)
            2. func must return a dataframe
        """
        pass
        # return self.func(target_df,*self.args, **self.kwargs)

    # def apply_init(self,target_df=None): 
    #     if self.init_func is not None:
    #         self.init_func(target_df,*self.args,**self.kwargs)

    def _apply_function_to(self, target, f):
        val = f(target, *self.args, **self.kwargs)
        if val == None: 
            return
        i_j = target._row_query,target._col_query

        # If the transform returns something, it should not be empty. 
        if val.shape[0] == 0 or val.shape[1] == 0:
            raise ValueError

        # TODO: Remove this code, and require the user to specify the reshaping
        # in the init function? 

        # If the target is empty, nothing depends on it yet and we can treat it
        # as a new partition

        # TODO: If the target is a matrix or an integer, we should handle that
        # as well. 

        # If the target is empty, then we can just set the value. 
        if target.empty():
            target.set_dataframe(val)
        # If the target is non-empty and shapes are wrong, then extend it
        elif val.shape != target.shape:
            for k_l in target._top_df._get_all_dependents(i_j):
                target._top_df._propogate_stop(k_l)

            # Extend the DF, set the value, and set to green.
            target._extend(val._row_index.keys(),val._col_index.keys())
            target.set_dataframe(val)
            target._top_df._graph.node[i_j]["status"] = target.STATUS_GREEN

            # Now restart all the rest
            for k_l in target._top_df._get_all_dependents(i_j):
                target._top_df._propogate_start(k_l,ignore=i_j)
        # If the target is non-empty but the value matches, then set the data
        else: 
            target.set_dataframe(val)

    # def apply_continuous(self, target):
    #     """ Apply a function continuously in a thread, and return the thread.
    #     """
    #     # Run at least once
    #     print "running continuously"
    #     thread = Thread(target = self._continuous_wrapper, args=(target,))
    #     thread.start()

    #     return thread



class BatchTransform(Transform):
    def apply(self, target_df):
        self._apply_function_to(target_df,self.func)

    @abstractmethod 
    def func(self, target_df, *args, **kwargs):
        pass


class ContinuousTransform(Transform):
    def apply(self, target_df):
        self.init_func(target_df, *self.args, **self.kwargs)
        thread = Thread(target = self._continuous_wrapper, args=(target_df,))
        thread.start()
        return thread

    @abstractmethod
    def init_func(self, target_df, *args, **kwargs):
        pass

    @abstractmethod
    def continuous_func(self, target_df, *args, **kwargs):
        pass

    def _continuous_wrapper(self, target_df):
        i_j = (target_df._row_query,target_df._col_query)
        graph = target_df._top_df._graph
        while(graph.node[i_j]["status"] is not target_df.STATUS_RED):
            # If this child is not blocked then we can run the function
            self._apply_function_to(target_df,self.continuous_func)

Transform.register(BatchTransform)
Transform.register(ContinuousTransform)

from bokeh.client import push_session
from bokeh.io import curdoc
from bokeh.embed import autoload_server

class FigureTransform(ContinuousTransform):
    def apply(self,target_df):
        self.init_func(target_df, *self.args, **self.kwargs)

        self.session = push_session(curdoc())
        tag = autoload_server(self.p,session_id=self.session.id)
        target_df._top_df._plots.append(tag)

        thread = Thread(target = self._continuous_wrapper, args=(target_df,))
        thread.start()
        return thread

    def init_func(self,target_df,*args,**kwargs):
        self.p = self.create_figure(target_df,*args,**kwargs)

    @abstractmethod
    def create_figure(self, target_df, *args, **kwargs):
        pass

    def continuous_func(self,target_df,*args,**kwargs):
        self.update(self.p)

    @abstractmethod
    def update(self, target_df, *args, **kwargs):
        pass