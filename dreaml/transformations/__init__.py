"""
Copyright 2015 Zico Kolter, Eric Wong, Terrence Wong

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__version__ = "0.0.1"
from dot import Dot
from gd import GD
from identity import Identity
from kitchensinks import KitchenSinks
from linear import Linear
from onehotencoding import OneHotEncoding
from pca import PCA
from pcabasis import PCABasis
from permute import Permute
from sgd import SGD
from zeromean import ZeroMean
from metrics import Metrics
from kmpp import KMPP