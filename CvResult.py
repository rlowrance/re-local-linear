import numpy as np

from FoldResult import FoldResult
import Maybe


class CvResult(object):
    '''Cross-validation result.

    Created by running main program cv-cell.py

    Contains results from n-fold cross validation.
    '''

    def __init__(self):
        self.fold_results = {}

    def save_FoldResult(self, fr):
        self.fold_results[len(self.fold_results)] = fr

    def median_errors_ignore_nans(self):
        '''Return Maybe(np.array of median errors from folds).
        '''
        results = []
        for fold_result in self.fold_results.itervalues():
            statistic = fold_result.median_error_ignore_nans()
            if statistic.has_value:
                results.append(statistic.value)

        if len(results) == 0:
            return Maybe.NoValue()
        else:
            return Maybe.Maybe(np.array(results))

if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):

        def setUp(self):
            pass

        def test_median_exists(self):
            fr1 = FoldResult()
            fr1.extend(np.array([1, 2]),
                       np.array([2, 3]))

            fr2 = FoldResult()
            fr2.extend(np.array([10, 20]),
                       np.array([20, 30]))

            cvresult = CvResult()
            cvresult.save_FoldResult(fr1)
            cvresult.save_FoldResult(fr2)

            median_values = cvresult.median_errors_ignore_nans()
            if median_values.has_value:
                median_value = np.median(median_values.value)
                self.assertAlmostEqual(median_value, -5.5)
                return
            else:
                self.fail()  # no value

        def test_median_doesnt_exist(self):
            fr = FoldResult()
            fr.extend(np.array([np.nan, 2]),
                      np.array([1,      np.nan]))

            cvresult = CvResult()
            cvresult.save_FoldResult(fr)

            median_values = cvresult.median_errors_ignore_nans()
            self.assertTrue(not median_values.has_value)

    unittest.main()
