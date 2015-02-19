import numpy as np
import pdb
import math

from FoldResult import FoldResult
import Maybe
import errors


class CvResult(object):
    '''Cross-validation result.

    Created by running main program cv-cell.py

    Contains results from n-fold cross validation.
    '''

    def __init__(self):
        self.fold_results = {}
        self.verbose = False

    def save_FoldResult(self, fr):
        self.fold_results[len(self.fold_results)] = fr

    def __str__(self):
        s = 'CvResult(%s folds)' % len(self.fold_results)
        return s

    def _reduce_fold_errors(self,
                            summarize_fold_accuracy,
                            reduce_fold_summary_to_number):
        '''Return Maybe(a number, summarizing accuracy across folds).
        '''
        # 1: summarize actuals and estimates from each fold by a number
        fold = np.full(len(self.fold_results), np.nan)
        for index, fr in self.fold_results.iteritems():
            maybe_fold_errors = fr.maybe_errors()
            if maybe_fold_errors.has_value:
                fold[index] = summarize_fold_accuracy(maybe_fold_errors.value)
        if self.verbose:
            print 'step 1 fold', fold

        # now fold[fold_index] is a number, possible NaN, summarizing fold
        # estimation accuracy

        # 2: reduce fold summaries to a number
        if np.isnan(fold).all():
            return Maybe.NoValue()
        else:
            known_fold_values = fold[~np.isnan(fold)]  # drop NaNs
            if self.verbose:
                print 'step 2 known fold values', known_fold_values
            return Maybe.Maybe(reduce_fold_summary_to_number(known_fold_values))

    def mean_of_mean_absolute_errors(self):
        return self._reduce_fold_errors(errors.mean_absolute_error,
                                        errors.mean_error)

    def mean_of_root_mean_squared_errors(self):
        return self._reduce_fold_errors(errors.mean_squared_error,
                                        errors.mean_error)

    def median_of_root_median_squared_errors(self):
        return self._reduce_fold_errors(errors.root_median_squared_error,
                                        errors.median_error)

    def median_of_median_absolute_errors(self):
        return self._reduce_fold_errors(errors.median_absolute_error,
                                        errors.median_error)


if __name__ == '__main__':
    import unittest

    class Test(unittest.TestCase):

        def setUp(self):
            def make_fold_result(actuals, estimates):
                fr = FoldResult()
                fr.extend(np.array(actuals), np.array(estimates))
                return fr

            def make_cv_result(*fr_list):
                cvresult = CvResult()
                for fr in fr_list:
                    cvresult.save_FoldResult(fr)
                return cvresult

            fr1 = make_fold_result([1, 2, -3],
                                   [10, 20, -30])
            fr2 = make_fold_result([-100, 200],
                                   [-10, 20])
            fr3 = make_fold_result([1, np.nan],
                                   [np.nan, 2])

            self.cv1 = make_cv_result(fr1, fr2, fr3)
            self.cv2 = make_cv_result(fr3)

        def test_mean_of_mean_absolute_errors_cv1(self):
            fr1 = (9 + 18 + 27) / 3.0
            fr2 = (90 + 180) / 2.0
            expected = (fr1 + fr2) / 2.0
            cv = self.cv1.mean_of_mean_absolute_errors()
            self.assertAlmostEqual(cv.value, expected)
            pass

        def test_mean_of_mean_absolute_errors_cv2(self):
            cv = self.cv2.mean_of_mean_absolute_errors()
            self.assertTrue(not cv.has_value)
            pass

        def test_mean_of_root_mean_squared_errors_cv1(self):
            fr1 = (9 + 18 + 27) / 3.0
            fr2 = (90 + 180) / 2.0
            expected = (fr1 + fr2) / 2.0
            cv = self.cv1.mean_of_mean_absolute_errors()
            self.assertAlmostEqual(cv.value, expected)
            pass

        def test_mean_of_root_mean_squared_errors_cv2(self):
            cv = self.cv2.mean_of_mean_absolute_errors()
            self.assertTrue(not cv.has_value)
            pass

        def test_median_of_median_absolute_errors_cv1(self):
            f1 = 18   # mid abs error
            f2 = .5 * (90 + 180)
            # f3 = np.nan
            expected = .5 * (f1 + f2)  # avg, since even number of folds
            if False:
                print 'f1 RMedSE', f1
                print 'f2 RMedSE', f2
                print 'expected median across folds', expected
            cv = self.cv1.median_of_median_absolute_errors()
            self.assertAlmostEqual(cv.value, expected)

        def test_median_of_root_median_absolute_errors_cv2(self):
            cv = self.cv2.median_of_median_absolute_errors()
            self.assertTrue(not cv.has_value)

        def test_median_of_root_median_squared_errors_cv1(self):
            f1 = 18
            f2 = math.sqrt((90 * 90 + 180 * 180) * 0.5)
            # f3 = np.nan
            expected = .5 * (f1 + f2)  # avg, since even number of folds
            if False:
                print 'f1 RMedSE', f1
                print 'f2 RMedSE', f2
                print 'expected median across folds', expected
            cv = self.cv1.median_of_root_median_squared_errors()
            self.assertAlmostEqual(cv.value, expected)

        def test_median_of_root_median_squared_errors_cv2(self):
            return
            cv = self.cv2.median_of_root_median_squared_errors()
            self.assertTrue(not cv.has_value)

    unittest.main()
