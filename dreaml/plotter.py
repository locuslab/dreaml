from dataframe.dataframe import DataFrame
from dataframe.transform import ContinuousTransform
from time import time,sleep
from random import shuffle

from bokeh.plotting import figure
from bokeh.io import output_server, cursession,curdoc,show,push,reset_output
from bokeh.embed import autoload_server
from requests.exceptions import ConnectionError
import numpy as np


class Plotter(ContinuousTransform):

    def init_func(self,target_df,f,title,legend,interval=1,**kwargs):
        self.connect_to_server()
        self.p = figure(plot_width=400, plot_height=400,title=title)
        (y0,x0) = f(**kwargs)
        target_df["x/","val/"].set_matrix(np.array(x0).reshape(1,len(x0)))
        target_df["y/","val/"].set_matrix(np.array(y0).reshape(1,len(y0)))

        for i in range(len(legend)):
            self.p.line([],[],name=legend[i])

        push()
        tag = autoload_server(self.p,cursession())
        target_df._top_df._plots.append(tag)


    def continuous_func(self,target_df,f,title,legend,interval=1,**kwargs):
        (y0,x0) = f(**kwargs)
        self.update(y0,x0,legend)
        sleep(interval)


    def connect_to_server(self):
        if cursession() == None:
            try: 
                output_server("dreaml")
            except ConnectionError: 
                reset_output()
                print "Failed to connect to bokeh server"

    def update(self,y0,x0,legend):
        assert(len(y0)==len(x0))
        for i in range(len(y0)):
            renderer = self.p.select(dict(name=legend[i]))
            ds = renderer[0].data_source
            ds.data["y"].append(y0[i])
            ds.data["x"].append(x0[i])
            cursession().store_objects(ds)