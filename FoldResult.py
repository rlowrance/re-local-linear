# system imports
import numpy as np

# local imports
import Maybe

class FoldResult(object):
    '''Computations on a cross validation fold.'''

    def __init__(self):
        self.actuals = np.array([])
        self.estimates = np.array([])

    def extend(self, actuals, estimates):
        '''Extend the collection of values.

        Args
        actuals   1D np.array
        estimates 1D np.array

        Returns: None
        '''
        if len(actuals) != len(estimates):
            print 'len(actuals)', len(actuals)
            print 'len(estimates)', len(estimates)
            raise ValueError('lengths differ')
        self.actuals = np.append(self.actuals, actuals)
        self.estimates = np.append(self.estimates, estimates)

    def reduce_erros_ignore_nans(self, reduction):
        '''Reduce the non-nan errors.'''
        errors = self.actuals - self.estimates
        if np.isnan(errors).all():
            # return NoValue, if
            # - errors.size == 0 OR
            # - every element of errors is nan
            # NOTE: The one test does it all
            return Maybe.NoValue()
        else:
            return Maybe.Maybe(reduction(errors))

    def mean_error_ignore_nans(self):
        '''Return Maybe(number), ignoring actuals or estimates with Nan or Inf.
        '''
        return self.reduce_erros_ignore_nans(np.nanmean)

    def median_error_ignore_nans(self):
        '''Return median error or NaN, ignoring actuals or estimates with NaN.
        '''
        return self.reduce_erros_ignore_nans(np.nanmedian)

if __name__ == '__main__':
    import unittest
    import numpy as np

    class Test(unittest.TestCase):

        def setUp(self):
            self.verbose = False
            self.group1_actuals = np.array([1, 2, np.nan, 4])
            self.group1_estimates = np.array([2, 3, 4, np.nan])

            self.group2_actuals = np.array([10])
            self.group2_estimates = np.array([20])

            # build the FoldResult object
            fr = FoldResult()
            fr.extend(self.group1_actuals,
                      self.group1_estimates)
            fr.extend(self.group2_actuals,
                      self.group2_estimates)
            self.fr = fr

        def test_extend_one(self):
            a = np.array([1, 2, 3])
            b = np.array([4, 5, 6])
            fr = FoldResult()
            fr.extend(a, b)
            if self.verbose:
                print fr.actuals
                print fr.estimates
            self.assertEqual(fr.actuals.size, 3)
            self.assertEqual(fr.actuals[1], 2)

        def test_extend_two(self):
            a = np.array([1, 2, 3])
            b = np.array([4, 5, 6])
            fr = FoldResult()
            fr.extend(a, b)
            fr.extend(np.array([10, 20]),
                      np.array([40, 50]))
            if self.verbose:
                print fr.actuals
                print fr.estimates
            self.assertEqual(fr.estimates.size, 5)
            self.assertEqual(fr.estimates[4], 50)

        def test_mean_exists(self):
            fr = FoldResult()
            fr.extend(np.array([1, 2, np.nan, 10, 20]),
                      np.array([2, 3, 4,      20, np.nan]))
            value = fr.mean_error_ignore_nans()
            expected = -(1 + 1 + 10) / 3.0
            self.assertTrue(value.has_value)
            self.assertAlmostEqual(value.value, expected)

        def test_mean_doesnt_exist(self):
            actuals = np.array([np.nan, 2])
            estimates = np.array([1, np.nan])
            fr = FoldResult()
            fr.extend(actuals, estimates)
            mean_error = fr.mean_error_ignore_nans()
            self.assertTrue(not mean_error.has_value)

        def test_median_exists(self):
            value = self.fr.median_error_ignore_nans()
            expected = -1
            self.assertTrue(value.has_value)
            self.assertAlmostEqual(value.value, expected)

        def test_median_doesnt_exists(self):
            fr = FoldResult()
            fr.extend(np.array([]), np.array([]))
            value = fr.median_error_ignore_nans()
            self.assertTrue(not value.has_value)

    unittest.main()
