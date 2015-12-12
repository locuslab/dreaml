Example
===============

On this page, you will find a complete working example demonstrating the use of
dreaml on a standard machine learning task: classifying digits in the MNIST
dataset. 

We recommend running this example from the IPython notebook, which is located in
the examples directory is also on our GitHub page here_. 

.. _here: https://github.com/locuslab/dreaml

.. code-block:: python

    import cPickle, gzip
    import numpy as np
    import dreaml as dm
    from dreaml.server import start
    from dreaml.loss import Softmax
    import dreaml.transformations as trans

    f = gzip.open('examples/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    X_train = train_set[0]
    y_train = train_set[1][:,None]
    X_test = valid_set[0]
    y_test = valid_set[1][:,None]

    // First we initialize the DataFrame
    df = dm.DataFrame()

    // This line starts the webserver and launches a web page
    start(df)

    // We load the initial data into the DataFrame. All data and the structure
    // of the DataFrame are visible from the web frontend. 
    df["data/train/", "input/raw/"] = dm.DataFrame.from_matrix(X_train)
    df["data/train/", "input/label/"] = dm.DataFrame.from_matrix(y_train)
    df["data/test/", "input/raw/"] = dm.DataFrame.from_matrix(X_test)
    df["data/test/", "input/label/"] = dm.DataFrame.from_matrix(y_test)

    // We generate features and place them in a sub directory of a features/ folder
    df["data/", "features/pca/"] = trans.PCA(df["data/train/", "input/raw/"], 
                                             df["data/","input/raw/"],
                                             num_bases=50)

    // We generate more features, placing them in a different subdirectory
    df["data/", "features/ks1/"] = trans.KitchenSinks(df["data/","features/pca/"],
                                                      num_features=1000)

    reg = 0.01
    batch_size=50

    // Here we start stochastic gradient descent. It will run continuously in
    // the background in a thread. 
    df["weights/", "features/"] = trans.SGD(Softmax,
                                          np.zeros((50,1000)),
                                          df["data/train/", "features/"],
                                          df["data/train/","input/label/"],
                                          batch_size=50,
                                          reg=reg)

    // We calculate some metrics to measure the performance of our model
    df["data/","metrics/"] = trans.Metrics([Softmax.f_vec, Softmax.err],
                                           df["weights/", "features/"],
                                           df["data/", "features/"],
                                           df["data/", "input/label/"],
                                           reg=reg,
                                           metrics_names=["SoftmaxLoss",
                                                          "MulticlassError"])

    // Next, we define functions that return tuples of values we wish to plot
    def softmax_average():
        metrics = df["data/","metrics/"].get_matrix()
        n = df["data/train/","metrics/"].shape()[0]

        niters = df["weights/", "features/"].T().niters
        return ([np.mean(metrics[0:n,0])],[niters])

    def traintest_average():
        metrics = df["data/","metrics/"].get_matrix()
        n = df["data/train/","metrics/"].shape()[0]
        niters = df["weights/", "features/"].T().niters
        return ([np.mean(metrics[0:n,1]),np.mean(metrics[n+1:,1])],[niters,niters])

    // Passing these to the plotter, these graphs are now visible via the web
    // frontend. These plotters are also running as threads. 
    df["plot/","loss/"] = dm.Plotter(softmax_average,"objective loss",["softmax"])
    df["plot/","err/"] = dm.Plotter(traintest_average,
                                    "train and test err",
                                    ["train","test"])

    // Without needing to restart or stop any currently running threads, we can
    // add new features. When this transformation is done, all dependent
    // transformation will stop and restart automatically. 
    df["data/", "features/ks2/"] = trans.KitchenSinks(df["data/","features/pca/"],
                                                      num_features=1000)

    // We can stop transformations at any time
    df["weights/", "features/"].stop()