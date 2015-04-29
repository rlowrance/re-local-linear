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
                # squash large value that will fail on error * error
                errors = maybe_fold_errors.value
                errors[np.abs(errors) > 1e100] = 1e100
                fold[index] = summarize_fold_accuracy(errors)

        # now fold[fold_index] is a number, possible NaN, summarizing fold
        # estimation accuracy

        # 2: reduce fold summaries to a number
        if np.isnan(fold).all():
            return Maybe.NoValue()
        else:
            known_fold_values = fold[~np.isnan(fold)]  # drop NaNs
            return Maybe.Maybe(reduce_fold_summary_to_number(known_fold_values))

    def mean_of_root_mean_squared_errors(self):
        # return Maybe(mean of root mean squared errors)
        return self._reduce_fold_errors(errors.root_mean_squared_error,
                                        errors.mean_error)

    def median_of_root_median_squared_errors(self):
        # return Mabye(median of root median squared errors)
        return self._reduce_fold_errors(errors.root_median_squared_error,
                                        errors.median_error)

    def mean_of_fraction_wi10(self):
        # return Maybe(mean of fraction of estimates within 10 percent)
        # approach
        # 1. Determine for each fold the fraction within 10 percent
        # 2. Take the mean of these fractions
        fold = np.full(len(self.fold_results), np.nan)
        for index, fr in self.fold_results.iteritems():
            abs_errors = np.abs(fr.actuals - fr.estimates)
            abs_errors_is_less = (abs_errors / fr.actuals) < 0.10
            fold[index] = sum(abs_errors_is_less * 1.0 / fr.actuals.size)

        result = np.mean(fold)
        return Maybe.Maybe(result)


if __name__ == '__main__':
    import unittest
    if False:
        pdb.set_trace()

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
            fr4 = make_fold_result([0],
                                   [1e200])

            self.cv1 = make_cv_result(fr1, fr2, fr3)
            self.cv2 = make_cv_result(fr3)
            self.cv3 = make_cv_result(fr4)

        def test_mean_of_root_mean_squared_errors_cv1(self):
            fr1 = math.sqrt((9 * 9 + 18 * 18 + 27 * 27) / 3.0)
            fr2 = math.sqrt((90 * 90 + 180 * 180) / 2.0)
            expected = (fr1 + fr2) / 2.0
            cv = self.cv1.mean_of_root_mean_squared_errors()
            self.assertAlmostEqual(cv.value, expected)
            pass

        def test_mean_of_root_mean_squared_errors_cv2(self):
            cv = self.cv2.mean_of_root_mean_squared_errors()
            self.assertTrue(not cv.has_value)
            pass

        def test_mean_of_root_mean_squared_errors_cv3(self):
            cv = self.cv3.mean_of_root_mean_squared_errors()
            self.assertTrue(cv.has_value)
            pass

        def test_median_of_root_median_squared_errors_cv1(self):
            f1 = math.sqrt(18 * 18)
            f2 = math.sqrt(.5 * (90 * 90 + 180 * 180))
            expected = .5 * (f1 + f2)
            cv = self.cv1.median_of_root_median_squared_errors()
            self.assertAlmostEqual(cv.value, expected)

        def test_median_of_root_median_squared_errors_cv2(self):
            cv = self.cv2.median_of_root_median_squared_errors()
            self.assertTrue(not cv.has_value)

    unittest.main()
