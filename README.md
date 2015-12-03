# dreaml
dreaml (dynamic reactive machine learning, procounced "Dream ML") is a Python library intended to help users quickly and interactively build, debug, and evaluate machine learning models.  As the same suggests, dreaml brings the concepts of functional reactive programming to machine learning, and is based upon two key ideas:
1. All data, features, parameters, and predictions are jointly stored in a single unified data structure, the hierarchical dynamic data frame (HDDF).  This data structure lets users store block matrix data with a hierarhical structure in both rows and columns.
2. All machine learning operations, from simple data transformation to complete runs of algorithms, are treated as transformations upon blocks in the HDDF.  These transformations run continuously and are responsive to changes, so that users can quickly add features, add data, adjust hyperparameter, etc, all while the the machine learning models update accordingly in an online fashion.

Documentation and examples are available at http://www.dreaml.io

# Installation
You can install the library using `pip install dreaml`.  

# Development
dreaml is under active development and we expect that some aspects of the system and API may change in the future.  Any feature requests, bug reports, or other comments are greatly appreciated.
