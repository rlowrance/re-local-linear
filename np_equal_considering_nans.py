def np_equal_considering_nans(a, b):
    '''
    Are two nparrays equal, except when both a NaN?

    Arguments
    ---------
    a : array_like
    b : array_like

    Returns
    -------
    boolean
    '''
    import numpy as np
    a_is_nan = np.isnan(a)
    b_is_nan = np.isnan(b)
    if np.array_equal(a_is_nan, b_is_nan):
        return np.array_equal(np.logical_not(a_is_nan),
                              np.logical_not(a_is_nan))
    else:
        return np.array_equal(a, b)

if __name__ == '__main__':
    import numpy as np
    import unittest

    class Test(unittest.TestCase):
        def test_equal_no_nans(self):
            a = np.float64([123, 456])
            b = a
            self.assertTrue(np_equal_considering_nans(a, b))

        def test_notequal_a_has_nan(self):
            a = np.float64([123, np.nan])
            b = np.float64([123, 456])
            self.assertFalse(np_equal_considering_nans(a, b))

        def test_notequal_b_has_nan(self):
            b = np.float64([123, np.nan])
            a = np.float64([123, 456])
            self.assertFalse(np_equal_considering_nans(a, b))

        def test_equal_both_have_nan(self):
            a = np.float64([123, np.nan])
            b = np.float64([123, np.nan])
            self.assertTrue(np_equal_considering_nans(a, b))

    unittest.main()
