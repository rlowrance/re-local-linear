# establish equivalence of rootMsquaredError and MAbsoluteError
# for M in {Mean, Median}

import errors
import numpy as np


n_tests = 10
n_dimensions = 3
tolerance = 1e-3


def root_median_squared_error(x):
    return np.sqrt(np.median(x * x))


def check(x, a, b):
    if abs(a - b) < tolerance:
        pass
    else:
        print x, a, b


def check_many(fa, fb):
    for test in xrange(n_tests):
        x = np.random.rand(n_dimensions)
        a = fa(x)
        b = fb(x)
        check(x, a, b)

print 'checking mean'
check_many(errors.root_mean_squared_error,
           errors.mean_absolute_error)

print 'checking median'
# equal if an odd dimension, not if an even dimension
check_many(root_median_squared_error,
           errors.absolute_median_error)
