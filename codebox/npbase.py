import os, sys, json
import math
from collections import namedtuple
import numpy as np
import pandas as pd
from numpy.random import RandomState
arr = np.array
pfloat = lambda x, rod=4: float(round(float(x), rod)) # python float
Stat = namedtuple('Stat', ['mean', 'std', 'min', 'max'])
stat = lambda x, rod=4: Stat(np.round(np.mean(x), rod), np.round(np.std(x), rod), np.round(np.min(x), rod), np.round(np.max(x), rod))
rnd = RandomState()
rnd1337 = RandomState(1337)
np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.max_colwidth', 810664)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f'%x)

# weighted shuffle
wshf = lambda u01, w: -u01**(1.0/w)
apply1d = lambda f, a: np.apply_along_axis(f, 0, a)

# example
'''
x = rnd1337.standard_t(10, size=[15])
w = apply1d(lambda v: wshf(rnd1337.uniform(size=v.shape), np.exp(v)), x)
i = np.argsort(w)
xi = x[i]
'''

