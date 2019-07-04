# This file holds global (DAPPER-wide) imports and settings
print("Initializing DAPPER...",end="", flush=True)

##################################
# Scientific
##################################
import numpy as np
import scipy as sp
import numpy.random
import scipy.linalg as sla
import numpy.linalg as nla
import scipy.stats as ss


from scipy.linalg import svd
from numpy.linalg import eig
# eig() of scipy.linalg necessitates using np.real_if_close().
from scipy.linalg import sqrtm, inv, eigh

from numpy import \
    pi, nan, \
    log, log10, exp, sin, cos, tan, \
    sqrt, floor, ceil, \
    mean, prod, \
    diff, cumsum, \
    array, asarray, asmatrix, \
    linspace, arange, reshape, \
    eye, zeros, ones, diag, trace \
# Don't shadow builtins: sum, max, abs, round, pow



##################################
# Tools
##################################
import sys
assert sys.version_info >= (3,5)
import os.path
from time import sleep
from collections import OrderedDict
import warnings
import traceback
import re
import functools

# Pandas changes numpy's error settings. Correct.
olderr = np.geterr()
import pandas as pd
np.seterr(**olderr)

# Profiling. Decorate the function you wish to time with 'profile' below
# Then launch program as: $ kernprof -l -v myprog.py
import builtins
try:
    profile = builtins.profile     # will exists if launched via kernprof
except AttributeError:
    def profile(func): return func # provide a pass-through version.


# Installation suggestions
def install_msg(package):
  return """
  Could not find (import) package '{0}'. Using fall-back.
  [But we recommend installing '{0}' (using pip or conda, etc...)
  to improve the functionality of DAPPER.]""".format(package)
def install_warn(import_err):
  name = import_err.args[0]
  #name = name.split('No module named ')[1]
  name = name.split("'")[1]
  warnings.warn(install_msg(name))



##################################
# Plotting settings
##################################
def user_is_patrick():
  import getpass
  return getpass.getuser() == 'pataan'

import matplotlib as mpl

# is_notebook 
try:
  __IPYTHON__
  from IPython import get_ipython
  is_notebook = 'zmq' in str(type(get_ipython())).lower()
except (NameError,ImportError):
  is_notebook = False

# Choose graphics backend.
if is_notebook:
  mpl.use('nbAgg') # interactive
else:
  # terminal frontent
  if user_is_patrick():
    from sys import platform
    if platform == 'darwin':
      mpl.use('MacOSX') # prettier, stable, fast (notable in LivePlot)
      #mpl.use('Qt4Agg') # deprecated

      # Has geometry(placement). Causes warning
      #mpl.use('TkAgg')  
      #import matplotlib.cbook
      #warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
    else:
      pass

# Get Matlab-like interface, and enable interactive plotting
import matplotlib.pyplot as plt 
plt.ion()

# Styles, e.g. 'fivethirtyeight', 'bmh', 'seaborn-darkgrid'
plt.style.use(['seaborn-darkgrid','tools/DAPPER.mplstyle'])



##################################
# Imports from DAPPER package
##################################
from tools.colors import *
from tools.utils import *
from tools.multiprocessing import *
from tools.math import *
from tools.chronos import *
from tools.stoch import *
from tools.series import *
from tools.matrices import *
from tools.randvars import *
from tools.viz import *
from tools.liveplotting import *
from tools.localization import *
from tools.convenience import *
from tools.data_management import *
from da_methods.stats import *
from da_methods.admin import *
from da_methods.da_methods import *


print("Done") # ... initializing DAPPER



