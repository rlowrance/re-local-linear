# errors.py
# group error metrics, all of the same API
# ARG   : a 1D np.array of numbers, non of which are NaN
# RETURN: a number (scalar)

import numpy as np
import math
import pdb


def absolute_median_error(errors):
    return np.abs(np.median(errors))


def mean_absolute_error(errors):
    return np.mean(np.abs(errors))


def mean_error(errors):
    return np.mean(errors)


def median_absolute_error(errors):
    return np.median(np.abs(errors))


def median_error(errors):
    return np.median(errors)


def root_mean_squared_error(errors):
    return math.sqrt(np.mean(errors * errors))


def root_median_squared_error(errors):
    return math.sqrt(np.median(errors * errors))


if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):
        def setUp(self):
            self.e1 = np.array([1, -3, 2])
            self.e2 = np.array([4, -1, 3, -2])

        def test_absolute_median_error(self):
            self.assertAlmostEqual(1, absolute_median_error(self.e1))
            self.assertAlmostEqual(1, absolute_median_error(self.e2))

        def test_mean_absolute_error(self):
            self.assertAlmostEqual(2, mean_absolute_error(self.e1))
            self.assertAlmostEqual(2.5, mean_absolute_error(self.e2))

        def test_mean_error(self):
            self.assertAlmostEqual(0, mean_error(self.e1))
            self.assertAlmostEqual(1, mean_error(self.e2))

        def test_median_absolute_error(self):
            self.assertAlmostEqual(2, median_absolute_error(self.e1))
            self.assertAlmostEqual(2.5, median_absolute_error(self.e2))

        def test_median_error(self):
            self.assertAlmostEqual(1, median_error(self.e1))
            self.assertAlmostEqual(1, median_error(self.e2))

        def test_root_mean_squared_error(self):
            v1 = math.sqrt((1 + 9 + 4) / 3.0)
            v2 = math.sqrt((16 + 1 + 9 + 4) / 4.0)
            self.assertAlmostEqual(v1, root_mean_squared_error(self.e1))
            self.assertAlmostEqual(v2, root_mean_squared_error(self.e2))

        def test_root_median_squared_error(self):
            v1 = math.sqrt(4)
            v2 = math.sqrt(.5 * (4 + 9))
            self.assertAlmostEqual(v1, root_median_squared_error(self.e1))
            self.assertAlmostEqual(v2, root_median_squared_error(self.e2))

    unittest.main()
