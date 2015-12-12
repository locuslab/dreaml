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

Caching
-------

In dreaml, all resulting matrices from calls to ``get_matrix()`` are cached, to
prevent unnecessary overhead when performing get and set  operations on blocks
that require piecing together various parts of the DataFrame. This is because
reading and writing from the DataFrame can get expensive when the underlying
partitions are split into many small pieces. 

To get optimal performance, the user must be aware of the concept of thrashing:
repeated requests to overlapping but non-identical regions of the DataFrame.
For example, the following code will result in a cache eviction, which will
write the cached matrix back to the DataFrame:

.. code-block:: python
    
    // The following query will be cached
    df["data/train/","raw/"].get_matrix()

    // Since the following query overlaps with the previous, the previous
    // query will be evicted from the cache. 
    df["data/","raw/"].get_matrix()

This can be somewhat circumvented when dealing with *static* matrices whose
values don't change: supplying a ``readonly`` flag to ``get_matrix`` will
allow overlapping matrices to remain in the cache.

.. code-block:: python
    
    df["data/train/","raw/"].get_matrix(readonly=True)

    // Will not evict the previous query! 
    df["data/","raw/"].get_matrix(readonly=True)

Methods
-------

.. autoclass:: dreaml.DataFrame
   :members: