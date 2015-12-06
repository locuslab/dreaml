Server
=========

Overview
--------

You see visualizations of your data frame (e.g. data frame structure,
computational graph, real-time plots) via a web frontend. The frontend by
default is hosted on ``http://localhost:5000``. 

Plotting
--------

In order to view real-time plots, you must have a Bokeh_ server running. Be sure
to start it before attempting create plots (e.g., with ``bokeh-start``). The
plotting interface is given by
the :ref:`Plotter transformation<plotter>`. 

.. _Bokeh: http://bokeh.pydata.org/en/latest/

Methods
-------

.. automodule:: dreaml.server
   :members: