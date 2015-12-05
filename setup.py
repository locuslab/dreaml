from setuptools import setuptools


setup(
      name='dreaml',
      version='0.0.1',
      author = 'Zico Kolter, Eric Wong, Terrence Wong',
      author_email = 'zkolter@cs.cmu.edu, ericwong@cs.cmu.edu, tw@andrew.cmu.edu', 
      packages = ['dreaml',
                  'dreaml.index',
                  'dreaml.dataframe',
                  'dreaml.transform',
                  'dreaml.front_end'],
      package_dir = {'dreaml': 'dreaml'},
      url="http://www.bitbucket.org",
      install_requires = ["networkx >= 1.9.1",
                          "sortedcontainers >= 0.9.6",
                          "numpy >= 1.9.2",
                          "scipy >= 0.15.1",
                          "pandas >= 0.16.2"],
      )

