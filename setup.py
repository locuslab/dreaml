from setuptools import setup


setup(
      name='dreaml',
      version='0.0.7',
      author = 'Zico Kolter, Eric Wong, Terrence Wong',
      author_email = 'zkolter@cs.cmu.edu, ericwong@cs.cmu.edu, tw@andrew.cmu.edu', 
      packages = ['dreaml',
                  'dreaml.dataframe',
                  'dreaml.loss',
                  'dreaml.transformations',
                  'dreaml.frontend'],
      package_dir = {'dreaml': 'dreaml'},
      url="http://www.github.com",
      install_requires = ["networkx >= 1.9.1",
                          "numpy >= 1.9.2",
                          "scipy >= 0.15.1",
                          "pandas >= 0.16.2",
                          "sortedcontainers >= 0.9.6",
                          "nose >= 1.3.7",
                          "coverage >= 4.0",
                          "flask-bootstrap >= 3.3.5.7",
                          "flask-nav >= 0.5"
                          "bokeh >= 0.11.0"],
      include_package_data = True,
      )

