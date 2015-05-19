# system imports
import datetime
import numpy as np
import pdb
import unittest

# local imports
import Maybe


class FoldResult(object):
    '''Computations on a cross validation fold.'''

    def __init__(self):
        # vectors of results
        self.actuals = np.array([])
        self.estimates = np.array([])
        # dictionaries of results
        self.fitted = {}
        self.predictor_names = {}
        self.num_test = {}
        self.num_train = {}

    def save_num_test(self, date, num_test):
        self.num_test[date] = num_test

    def get_num_test(self):
        return self.num_test

    def save_num_train(self, date, num_train):
        self.num_train[date] = num_train

    def get_num_train(self):
        return self.num_train

    def save_fitted(self, date, fitted):
        self.fitted[date] = fitted

    def get_fitted(self):
        return self.fitted

    def save_predictor_names(self, date, predictor_names):
        self.predictor_names[date] = predictor_names

    def get_predictor_names(self):
        return self.predictor_names

    def extend(self, actuals, estimates):
        '''Extend the collection of values.

        Args
        actuals      1D np.array
        estimates    1D np.array

        Returns: None
        '''
        if len(actuals) != len(estimates):
            print 'len(actuals)', len(actuals)
            print 'len(estimates)', len(estimates)
            raise ValueError('lengths differ')
        self.actuals = np.append(self.actuals, actuals)
        self.estimates = np.append(self.estimates, estimates)

    def maybe_errors(self):
        '''Return Maybe(vector of errors, without nans).'''
        errors = self.actuals - self.estimates
        if np.isnan(errors).all():
            # return NoValue, if
            # - errors.size == 0 OR
            # - every element of errors is nan
            # NOTE: The one test does it all
            return Maybe.NoValue()
        else:
            errors_without_nans = errors[~np.isnan(errors)]
            return Maybe.Maybe(errors_without_nans)


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

    def test_fitted(self):
        # test methods .save_fitted() and .get_fitted()
        fr = self.fr
        fitted1 = ['a', 'b']
        date1 = datetime.date(2015, 05, 18)
        fr.save_fitted(date1, fitted1)
        fitted2 = ['aa', 'bb']
        date2 = datetime.date(2015, 05, 19)
        fr.save_fitted(date2, fitted2)
        fitted = fr.get_fitted()  # return a dict
        self.assertEqual(len(fitted), 2)

    def test_predictor_names(self):
        # test methods .save_preictor_names() and .get_predictor_names()
        fr = self.fr
        names1 = ['a', 'b']
        date1 = datetime.date(2015, 05, 18)
        fr.save_predictor_names(date1, names1)
        names2 = ['aa']
        date2 = datetime.date(2015, 05, 19)
        fr.save_predictor_names(date2, names2)
        predictor_names = fr.get_predictor_names()
        self.assertEqual(len(predictor_names), 2)

    def test_num_test(self):
        fr = self.fr
        fr.save_num_test(datetime.date(2015, 05, 18), 10)
        fr.save_num_test(datetime.date(2015, 05, 19), 20)
        num_tests = fr.get_num_test()
        self.assertEqual(len(num_tests), 2)

    def test_num_train(self):
        fr = self.fr
        fr.save_num_train(datetime.date(2015, 05, 18), 10)
        fr.save_num_train(datetime.date(2015, 05, 19), 20)
        num_trains = fr.get_num_train()
        self.assertEqual(len(num_trains), 2)

    def test_has_elements(self):
        me = self.fr.maybe_errors()
        self.assertTrue(me.has_value)
        value = me.value
        expected_errors = np.array([-1, -1, -10])
        for i in xrange(3):
            self.assertEqual(value[i], expected_errors[i])

    def test_has_no_elements(self):
        fr = FoldResult()
        fr.extend(np.array([1, np.nan]),
                  np.array([np.nan, 2]))
        me = fr.maybe_errors()
        self.assertTrue(not me.has_value)

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


if __name__ == '__main__':
    import unittest

    if False:
        pdb.set_trace()  # avoid warning

    unittest.main()
