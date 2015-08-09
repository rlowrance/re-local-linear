'''examples for numpy and pandas'''

import numpy as np
import pandas as pd

# 1D numpy arrays
v = np.array([1, 2, 3], dtype=np.float64)  # also: np.int64
v.shape  # tuple of array dimensions
v.ndim   # number of dimensions
v.size   # number of elements
for elem in np.nditer(v):  # read-only iteration
    pass
for elem in np.nditer(v, op_flags='readwrite'):  # mutation iteration
    elem[...] = abs(elem)   # elipses is required
for elems in np.nditer(v, flags=['external_loop']):  # iterate i chunks
    print elems  # elems is a 1D vector
# basic indexing (using a slice or integer) ALWAYS generates a view
v[0:v.size:1]  # start:stop:step
v[10]
v[...]
# advanced indexing (using an ndarray) ALWAYS generates a copy
# advanced indexes are always broadcast
v[np.array([1, 2])]  # return new 1D with 2 elements
v[~np.isnan(v)]      # return new 1D with v.size elements

# pd.Index
# data: aray-like 1D of hashable items
# dtype: np.dtype
# copy: bool default ?
# name: obj, documentation
# tupleize_cols: bool, default True; if True, attempt to create MultiIndex
i = pd.Index(data, dtype, copy, name, tupleize_cols)
i.shape
i.ndim
i.size
i.values  # underlying data as ndarray
# generally don't apply methods directly to Index objects

# pd.Series
# data: array-like, dict, scalar
# index: array-like, index
# dtype: numpy.dtype
# copy: default False (True forces a copy of data)
s = pd.Series(data, index, dtype, name, copy)
s.values  # return ndarray
s.shape
s.ndim
s.size
# indexing and iteration
s.get(key[,default])  # key: label
s.loc[key]    # key: single label, list or array of labels, slice with labels, bool array
s.iloc[key]   # key: int, list or array of int, slice with ints, boolean array
s.iteritems()  # iterate over (index, value) pairs

# pd.DataFrame:
# data: numpy ndarray, dict, DataFrame
# index: index or array-like
# columns: index or array-like
# dtype: nparray dtype
# copy: boolean default False
df = pd.DataFrame(data, index, columns, dtype, copy)
df.shape
df.ndim
df.size
df.as_matrix([columns])  # convert to numpy array
df.loc[key]   # key: single lable, list or array of labels, slice of labels, bool array
df.iloc[key]  # key: int, list or array of int, slice of int, bool array
