Transformations
===============

Overview
--------

One of the powerful aspects of dreaml is that **any** arbitrary data
transformation can be defined to generate new blocks in the DataFrame. These
transformations are entirely modular: you can pick and choose transformations,
or define your own and replace existing transformations. 

We are always looking to grow the library of general built-in transformations!
Go ahead and submit a request on GitHub_ to add your own transformations to the
library.

.. _Github: https://github.com/locuslab/dreaml

User-defined Transformations
----------------------------
It is extremely simple to define your own transformation. As an example, here is
an implementation of a transformation that computes a standard projection into a
lower dimensional space via PCA:

.. code-block:: python

    class PCA(BatchTransform):
        def func(self,target_df,X_pca_df, X_full_df, num_bases=50):
            X_full = X_full_df.get_matrix()
            X_pca = X_pca_df.get_matrix()
            X_mean = np.mean(X_pca,axis=0)
            X_std = np.std(X_pca,axis=0)
            _,s,v_T = la.svd((X_pca - X_mean) / X_std)
            target_df.set_matrix(((X_full - X_mean) / X_std).dot(v_T.T[:,:numbases])))

There are currently two types of transforms: ``BatchTransform`` and
``ContinuousTransform``. To create a new transform, you create a class which
inherits one of these two types, and fill in the function signature needed for
that type. 

Batch Transformations
---------------------
A batch transformation is the simplest transformation that runs exactly once. 

It is very simple to define: all you have to do is define ``func``, which takes
in as arguments the target DataFrame and any other arguments for the
transformation, performs whatever task you want, and (typically) saves the
result in the target DataFrame:

.. code-block:: python

    // The BatchTransform signature
    from dreaml.dataframe.transform import BatchTransform

    class YourBatchTransformation(BatchTransform):

        def func(self, target_df, *args, **kwargs):
            // implement code here to calculate some result M, and
            // typically save the answer with target_df.set_matrix(M)

As another example, here is a simple dot product transformation, which retains
the labels of the inputs to the target: 

.. code-block:: python

    from dreaml.dataframe.transform import BatchTransform

    class Dot(BatchTransform):
        """ Calculates the dot product of the contents of X_df and y_df """

        def func(self,target_df,X_df,Y_df):
            x = X_df.get_matrix()
            y = Y_df.get_matrix()
            row_labels = X_df._row_index.keys()
            col_labels = Y_df._col_index.keys()
            target_df.set_matrix(x.dot(y),row_labels,col_labels)

Continuous Transformations
--------------------------
A continuous transformation is a tranformation that applies some function
repeatedly. This function is run asynchronously within a Python thread. 

Defining a continuous transformation is very similar to a batch transformation.
You need to define two functions: ``init_func`` which will be run exactly once,
and ``continuous_func`` which will be run repeatedly. Both functions take as
arguments the target dataframe, and the same set of user supplied arguments. 

.. code-block:: python

    // The ContinuousTransform signature
    class YourContinuousTransform(ContinuousTransform):
        
        def init_func(self, target_df, *args, **kwargs):
            // Your initialization code here

        def continuous_func(self, target_df, *args, **kwargs):
            // Your continuously running code here

Built-in Transformations
------------------------
This section documents a list of transformations that are currently built-in to
the dreaml system.

.. automodule:: dreaml.transformations
   :members: