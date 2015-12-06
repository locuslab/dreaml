DataFrame
=========

Overview
--------

The DataFrame class is the underlying data structure for all dreaml operations.
The main method of accessing data within the dataframe is via a hierarchical
index into both the row and column indices. 

For example, we can get and set parts of the dataframe as follows: 

.. code-block:: python

    import dreaml as dm
    df = dm.DataFrame()
    df["data/train/","raw/"].set_matrix(X_train)
    df["data/train/","labels/"].set_matrix(y_train)
    df["data/test/","raw/"].set_matrix(X_test)
    df["data/test/","labels/"].set_matrix(y_test)

    // Is equivalent to np.vstack([X_train,X_test])
    df["data/","raw/"].get_matrix()

Consequently, ``df["data/","raw/"]`` is a data frame containing both the
training and testing raw data, and ``df["data/","labels/"]`` contains both the
training and testing labels. 

Methods
-------

.. autoclass:: dreaml.DataFrame
   :members: