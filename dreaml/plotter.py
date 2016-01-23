from dataframe.dataframe import DataFrame
from dataframe.transform import ContinuousTransform
from time import time,sleep
from random import shuffle, randint

from bokeh.plotting import figure
from bokeh.client import pull_session
from bokeh.io import output_server,curdoc,reset_output
from bokeh.embed import autoload_server
from requests.exceptions import ConnectionError
import numpy as np


class Plotter(ContinuousTransform):
    """Calculates ``f(**kwargs)`` at regular intervals and pushes the resulting
    plot to the web frontend. f must return ``(ys,xs)`` where  ``ys`` is the
    is a list of outputs, each of which is plotted with the corresponding
    ``xs`` entry on the same plot. """

    def __init__(self,f,title,legend=None,interval=1,colors=None,**kwargs):
        if legend==None:
            legend = []
        if colors==None:
            colors=[]
        super(Plotter,self).__init__(f,title,legend=legend,colors=colors,interval=1,**kwargs)

    def init_func(self,target_df,f,title,
                  legend=None,interval=1,colors=None,**kwargs):
        self.connect_to_server()
        self.p = figure(plot_width=400, plot_height=400,title=title)
        (y0,x0) = f(**kwargs)
        target_df["x/","val/"].set_matrix(np.array(x0).reshape(1,len(x0)))
        target_df["y/","val/"].set_matrix(np.array(y0).reshape(1,len(y0)))

        if len(legend)==0:
            for i in range(len(y0)):
                legend.append(str(i))
        if len(colors)==0:
            for i in range(len(y0)):
                color = "#%06x" % randint(0, 0xFFFFFF)
                colors.append(color)

        if len(y0)!= len(x0):
            raise ValueError("f must return two lists of equal length, equal\
                to the number of lines to be plotted.")
        if len(y0)!= len(legend):
            raise ValueError("length of legend must match number of\
                lines plotted")
        if len(y0) != len(colors):
            raise ValueError("length of list of colors must much number of\
                lines plotted")

        for i in range(len(legend)):
            self.p.line([],
                        [],
                        name=legend[i],
                        legend=legend[i],
                        color=colors[i])

        self.session = push_session(self.p)
        tag = autoload_server(self.p)
        target_df._top_df._plots.append(tag)


    def continuous_func(self,target_df,f,title,
                        legend=[],interval=1,colors=[],**kwargs):
        (y0,x0) = f(**kwargs)
        self.update(y0,x0,legend)
        sleep(interval)


    # def connect_to_server(self):
    #     try: 
    #         push_session
    #     except ConnectionError: 
    #         reset_output()
    #         print "Failed to connect to bokeh server"

    def update(self,y0,x0,legend):
        assert(len(y0)==len(x0))
        for i in range(len(y0)):
            renderer = self.p.select(dict(name=legend[i]))
            ds = renderer[0].data_source
            ds.data["y"].append(y0[i])
            ds.data["x"].append(x0[i])
            self.session.store_objects(ds)