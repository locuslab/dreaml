dreaml: dynamic reactive machine learning
=========================================

Overview
========
dreaml is a programming framework that brings the power of functional reactive
programming to the frontline of machine learning problems. The standard
application of machine learning occurs in a train-wait-adjust loop, where
practitioners start training, wait some amount of time, adjust their model, and
restart. The dreaml framework instead utilizes the **reactive** paradigm:
adjustments and changes to the model are immediately visible to the user.

The dreaml framework represents all operations as arbitrary data
transformations upon parts of a hierarchical dynamic data frame (HDDF). By
combining reactive principles with user-defined transformations in the HDDF,
any type of machine learning problem can be ported to dreaml for an
interactive, online model building experience. 

Contents:

.. toctree::
   :maxdepth: 2

   dataframe
   server 
   transformations
   plotter
   loss


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

