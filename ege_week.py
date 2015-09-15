'''create files contains estimated generalization errors for model

INPUT FILE
 WORKING/transactions-subset2.pickle

OUTPUT FILES
 WORKING/ege_week/YYYY-MM-DD/MODEL-TD/HP-FOLD.pickle  dict all_results
 WORKING/ege_month/YYYY-MM-DD/MODEL-TD/HP-FOLD.pickle  dict all_results
'''

import collections
import cPickle as pickle
import datetime
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import ensemble
import sys
import warnings

from Bunch import Bunch
from DataframeAppender import DataframeAppender
from directory import directory
from Logger import Logger
import parse_command_line


def usage(msg=None):
    if msg is not None:
        print 'invocation error: ' + str(msg)
    print 'usage: python ege_week.py YYYY-MM-DD <other options>'
    print ' YYYY-MM-DD       mid-point of week; analyze -3 to +3 days'
    print ' --month          optional; test on next month, not next week'
    print ' --model {lr|rf}  which model to run'
    print ' --td <range>     training_days'
    print ' --hpd <range>    required iff model is rf; max_depths to model'
    print ' --hpw <range>    required iff model is rf; weight functions to model'
    print ' --hpx <form>     required iff mode is lr; transformation to x'
    print ' --hpy <form>     required iff mode is lr; transformation to y'
    print ' --test           optional; if present, program runs in test mode'
    print 'where'
    print ' <form>  is {lin|log}+ saying whether the variable is in natural or log units'
    print ' <range> is start [stop [step]], just like Python\'s range(start,stop,step)'
    sys.exit(1)


DateRange = collections.namedtuple('DateRange', 'first last')


def make_DateRange(mid, half_range):
    return DateRange(first=mid - datetime.timedelta(half_range),
                     last=mid + datetime.timedelta(half_range),
                     )




def make_predictors():
    '''return dict key: column name, value: whether and how to convert to log domain

    Include only features of the census and tax roll, not the assessment,
    because previous work found that using features derived from the
    assessment degraded estimated generalization errors.

    NOTE: the columns in the x_array objects passed to scikit learn are in
    this order. FIXME: that must be wrong, as we return a dictionary
    '''
    # earlier version returned a dictionary, which invalided the assumption
    # about column order in x
    result = (  # the columns in the x_arrays are in this order
        ('fraction.owner.occupied', None),
        ('FIREPLACE.NUMBER', 'log1p'),
        ('BEDROOMS', 'log1p'),
        ('BASEMENT.SQUARE.FEET', 'log1p'),
        ('LAND.SQUARE.FOOTAGE', 'log'),
        ('zip5.has.industry', None),
        ('census.tract.has.industry', None),
        ('census.tract.has.park', None),
        ('STORIES.NUMBER', 'log1p'),
        ('census.tract.has.school', None),
        ('TOTAL.BATHS.CALCULATED', 'log1p'),
        ('median.household.income', 'log'),  # not log feature in earlier version
        ('LIVING.SQUARE.FEET', 'log'),
        ('has.pool', None),
        ('zip5.has.retail', None),
        ('census.tract.has.retail', None),
        ('is.new.construction', None),
        ('avg.commute', None),
        ('zip5.has.park', None),
        ('PARKING.SPACES', 'log1p'),
        ('zip5.has.school', None),
        ('TOTAL.ROOMS', 'log1p'),
        ('age', None),
        ('age2', None),
        ('effective.age', None),
        ('effective.age2', None),
    )
    return result


class CensusAdjacencies(object):
    def __init__(self):
        path = directory('working') + 'census_tract_adjacent.pickle'
        f = open(path, 'rb')
        self.adjacent = pickle.load(f)
        f.close()

    def adjacen(self, census_tract):
        return self.adjacent.get(census_tract, None)


def make_weights_function_1(train_df, census_adjacencies):
    'return function(df_row) -> vector of weights'
    train = train_df.copy(deep=True)
    adjacencies = census_adjacencies.copy()
    if True:
        print 'shape train df', train_df.shape
        print 'len adjacencies', len(adjacencies)

    def weights(df_row):
        'return vector of weights relative to the training data'
        r = np.ones(len(train))  # TODO: use the census tracts
        return r

    return weights


ModelId = collections.namedtuple('ModelId', 'name instance training_days hp')


def make_control(argv):
    'Return control Bunch'''

    print 'argv'
    pprint(argv)

    if len(argv) < 3:
        usage('missing invocation options')

    def make_sale_date(s):
        year, month, day = s.split('-')
        return datetime.date(int(year), int(month), int(day))

    pcl = parse_command_line.ParseCommandLine(argv)
    arg = Bunch(
        base_name=argv[0].split('.')[0],
        hpd=pcl.get_range('--hpd') if pcl.has_arg('--hpd') else None,
        hpw=pcl.get_range('--hpw') if pcl.has_arg('--hpw') else None,
        hpx=pcl.get_arg('--hpx') if pcl.has_arg('--hpx') else None,
        hpy=pcl.get_arg('--hpy') if pcl.has_arg('--hpy') else None,
        model=pcl.get_arg('--model'),
        month=pcl.has_arg('--month'),
        sale_date=make_sale_date(argv[1]),
        td=pcl.get_range('--td'),
        test=pcl.has_arg('--test'),
    )
    print 'arg'
    print arg
    # check for missing options
    if arg.model is None:
        usage('missing --model')
    if arg.td is None:
        usage('missing --td')

    # validate combinations of invocation options
    if arg.model == 'lr':
        if arg.hpx is None or arg.hpy is None:
            usage('model lr requires --hpx and --hpy')
    elif arg.model == 'rf':
        if arg.hpd is None or arg.hpw is None:
            usage('model rf requires --hpd and --hpw')
    else:
        usage('bad --model: %s' % str(arg.model))

    random_seed = 123
    now = datetime.datetime.now()
    predictors = make_predictors()
    print 'number of predictors', len(predictors)
    sale_date_range = make_DateRange(arg.sale_date, 15 if arg.month else 3)
    log_file_name = arg.base_name + '.' + now.isoformat('T') + '.log'
    # dir_out: WORKING/ege_[month|week]/<sale_date>/
    dir_out = (directory('working') +
               'ege_' +
               ('month' if arg.month else 'week') +
               '/' + argv[1] + '/'
               )

    debug = False
    test = arg.test

    b = Bunch(
        arg=arg,
        census_adjacencies=CensusAdjacencies(),
        date_column='python.sale_date',
        debug=debug,
        dir_out=dir_out,
        n_folds=2 if test else 10,
        n_rf_estimators=100 if test else 1000,  # num trees in a random forest
        path_in_old=directory('working') + 'transactions-subset2.pickle',
        path_in=directory('working') + 'transactions-subset3-subset-train.csv',
        path_log=directory('log') + log_file_name,
        predictors=predictors,
        price_column='SALE.AMOUNT',
        random_seed=random_seed,
        relevant_date_range=DateRange(first=datetime.date(2003, 1, 1), last=datetime.date(2009, 3, 31)),
        sale_date_range=sale_date_range,
        start_time=now,
        test=test,
        use_old_input=True,
    )
    return b


def elapsed_time(start_time):
    return datetime.datetime.now() - start_time


def x(mode, df, predictors):
    '''return 2D np.array, with df x values possibly transformed to log

    RETURNS array: np.array 2D
    '''
    def transform(v, mode, transformation):
        if mode is None:
            return v
        if mode == 'linear' or mode == 'lin':
            return v
        if mode == 'log':
            if transformation is None:
                return v
            if transformation == 'log':
                return np.log(v)
            if transformation == 'log1p':
                return np.log1p(v)
            raise RuntimeError('bad transformation: ' + str(transformation))
        raise RuntimeError('bad mode:' + str(mode))

    array = np.empty(shape=(df.shape[0], len(predictors)),
                     dtype=np.float64).T
    # build up in transposed form
    index = 0
    for predictor_name, transformation in predictors:
        v = transform(df[predictor_name].values, mode, transformation)
        array[index] = v
        index += 1
    return array.T


def y(mode, df, price_column):
    '''return np.array 1D with transformed price column from df'''
    df2 = df.copy(deep=True)
    if mode == 'log':
        df2[price_column] = pd.Series(np.log(df[price_column]), index=df.index)
    array = np.array(df2[price_column].as_matrix(), np.float64)
    return array


def demode(v, mode):
    'convert log domain to normal'
    if v is None:
        return None
    result = np.exp(v) if mode == 'log' else v
    return result


def errors(model_result):
    'return median_absolute_error and median_relative_absolute_error'
    actuals = model_result['actuals']
    estimates = model_result['estimates']
    abs_error = np.abs(actuals - estimates)
    median_abs_error = np.median(abs_error)
    rel_abs_error = abs_error / actuals
    median_rel_abs_error = np.median(rel_abs_error)
    return median_abs_error, median_rel_abs_error


class ReportLr(object):
    'report generation with y_mode and x_mode in key'
    # NOTE: perhaps reusable for any model with y and x modes

    def __init__(self):
        self.format_global_fold = '%10s %2d %3s %6s %3s %3s f%d %6.0f %3.2f'
        self.format_zip_fold = '%10s %2d %3s %6d %3s %3s f%d %6.0f %3.2f'
        self.format_global = '%10s %2d %3s %6s %3s %3s median %6.0f %3.2f'
        self.format_zip = '%10s %2d %3s %6d %3s %3s median %6.0f %3.2f'

    def global_fold_lines(self, key, result):
        fold_number, sale_date, training_days, model_name, scope = key
        assert(scope == 'global')
        for result_key, result_value in result.iteritems():
            y_mode = result_key[1][:3]
            x_mode = result_key[3][:3]
            median_abs_error, median_rel_abs_error = errors(result_value)
            line = self.format_global_fold % (sale_date,
                                              training_days,
                                              model_name,
                                              scope,
                                              y_mode,
                                              x_mode,
                                              fold_number,
                                              median_abs_error,
                                              median_rel_abs_error)
            yield line

    def zip_fold_lines(self, key, result):
        fold_number, sale_date, training_days, model_name, scope = key
        assert(isinstance(scope, tuple))
        assert(scope[0] == 'zip')
        zip_code = scope[1]
        for result_key, result_value in result.iteritems():
            y_mode = result_key[1][:3]
            x_mode = result_key[3][:3]
            median_abs_error, median_rel_abs_error = errors(result_value)

            line = self.format_zip_fold % (sale_date,
                                           training_days,
                                           model_name,
                                           zip_code,
                                           y_mode,
                                           x_mode,
                                           fold_number,
                                           median_abs_error,
                                           median_rel_abs_error)
            yield line

    def summarize_global(self,
                         sale_date,
                         training_days,
                         model_name,
                         all_results,
                         control):
        scope = 'global'
        for y_mode in ('log', 'linear'):
            y_mode_print = y_mode[:3]
            for x_mode in ('log', 'linear'):
                x_mode_print = x_mode[:3]
                median_errors = np.zeros(control.n_folds, dtype=np.float64)
                median_rel_errors = np.zeros(control.n_folds, dtype=np.float64)
                for fold_number in xrange(control.n_folds):
                    # determine errors in the fold
                    key = (fold_number, sale_date, training_days, model_name, scope)
                    if key not in all_results:
                        print 'key', key
                        print 'not in result'
                        continue
                    result = all_results[key]
                    model_result = result[('y_mode', y_mode, 'x_mode', x_mode)]
                    median_abs_error, median_rel_abs_error = errors(model_result)
                    fold_line = self.format_global_fold % (sale_date,
                                                           training_days,
                                                           model_name,
                                                           scope,
                                                           y_mode_print,
                                                           x_mode_print,
                                                           fold_number,
                                                           median_abs_error,
                                                           median_rel_abs_error)
                    print fold_line
                    median_errors[fold_number] = median_abs_error
                    median_rel_errors[fold_number] = median_rel_abs_error
                all_folds_line = self.format_global % (sale_date,
                                                       training_days,
                                                       model_name,
                                                       scope,
                                                       y_mode_print,
                                                       x_mode_print,
                                                       np.median(median_errors),
                                                       np.median(median_rel_errors))
                print all_folds_line

    def summarize_zip(self, sale_date, training_days, model_name,
                      all_results, control):

        def list_median(lst):
            assert(len(lst) > 0)
            return np.median(np.array(lst, dtype=np.float64))

        def report_zip_code(zip_code, keys):
            for y_mode in ('log', 'linear'):
                y_mode_print = y_mode[:3]
                for x_mode in ('log', 'linear'):
                    x_mode_print = x_mode[:3]
                    mode_key = ('y_mode', y_mode, 'x_mode', x_mode)
                    median_abs_errors = []
                    median_rel_abs_errors = []
                    for key in keys:
                        model_result = all_results[key][mode_key]
                        median_abs_error, median_rel_abs_error = errors(model_result)
                        fold_line = self.format_zip_fold % (sale_date,
                                                            training_days,
                                                            model_name,
                                                            zip_code,
                                                            y_mode_print,
                                                            x_mode_print,
                                                            key[0],  # fold number
                                                            median_abs_error,
                                                            median_rel_abs_error)
                        print fold_line
                        median_abs_errors.append(median_abs_error)
                        median_rel_abs_errors.append(median_rel_abs_error)
                    all_folds_line = self.format_zip % (sale_date,
                                                        training_days,
                                                        model_name,
                                                        zip_code,
                                                        y_mode_print,
                                                        x_mode_print,
                                                        list_median(median_abs_errors),
                                                        list_median(median_rel_abs_errors))
                    print all_folds_line

        # determine all zip codes in the specified lines
        zip_codes = collections.defaultdict(set)
        for key in all_results.keys():
            key_fold_number, key_sale_date, key_training_days, key_model_name, key_scope = key
            if key_scope == 'global':
                # examine only zip code scopes
                continue
            if key_sale_date == sale_date and key_training_days == training_days and key_model_name == model_name:
                key_zip_code = key_scope[1]
                zip_codes[key_zip_code].add(key)

        # process each zip code
        for zip_code, keys in zip_codes.iteritems():
            report_zip_code(zip_code, keys)

    def summarize(self, sale_date, training_days, model_name,
                  all_results, control):
        self.summarize_global(sale_date, training_days, model_name,
                              all_results, control)
        self.summarize_zip(sale_date, training_days, model_name,
                           all_results, control)


class Lr(object):
    'linear regression via sklearn'
    def __init__(self):
        self.Model_Constructor = linear_model.LinearRegression

    def reporter(self):
        return ReportLr

    def run(self, df_train, df_test, df_next, control):
        '''fit on training data and test

        ARGS
        train  : dataframe
        test   : dataframe
        control: Bunch

        RETURN dict of values
        dict key = (x_mode, y_mode)
             values = dict with keys 'actuals', 'estimates', 'fitted', x_names
        '''
        # implement variants
        verbose = False

        def variant(x_mode, y_mode):
            train_x = x(x_mode, df_train, control)
            test_x = x(x_mode, df_test, control)
            train_y = y(y_mode, df_train, control)
            model = self.Model_Constructor(fit_intercept=True,
                                           normalize=True,
                                           copy_X=True)
            fitted_model = model.fit(train_x, train_y)
            # if the model cannot be fitted, LinearRegressor returns
            # the mean of the train_y values
            estimates = fitted_model.predict(test_x)
            value = {
                'coef': fitted_model.coef_,
                'intercept_': fitted_model.intercept_,
                'estimates': demode(estimates, y_mode),
                'actuals': y('linear', df_test, control),
                'estimates_next': fitted_model.predict(x(x_mode, df_next, control)),
                'actuals_next': y(y_mode, df_next, control),
                'n_train': len(train_x),
            }
            # check results
            if verbose:
                print 'x_mode, y_mode: ', x_mode, y_mode
                print 'actuals: ', value['actuals']
                print 'estimates: ', value['estimates']
            return value

        all_variants = {}
        for x_mode in ('log', 'linear'):
            for y_mode in ('log', 'linear'):
                variant_value = variant(x_mode, y_mode)
                key = ('y_mode', y_mode, 'x_mode', x_mode)
                all_variants[key] = variant_value
        return all_variants


class ReportRf(object):
    'report generation w no variants (for now'

    def __init__(self):
        'sale_date days model global fold error abs_error'
        self.format_global_fold = '%10s %2d %3s %6s f%d maxdepth %2d %6.0f %3.2f'
        self.format_zip_fold = '%10s %2d %3s %6d f%d maxdepth %2d %6.0f %3.2f'
        self.format_global = '%10s %2d %3s %6s  median %6.0f %3.2f'
        self.format_zip = '%10s %2d %3s %6d  median %6.0f %3.2f'

    def global_fold_lines(self, key, result):
        fold_number, sale_date, training_days, model_name, scope = key
        assert(scope == 'global')
        for result_key, result_value in result.iteritems():
            assert result_key[0] == 'max_depth'
            max_depth = result_key[1]
            median_abs_error, median_rel_abs_error = errors(result_value)
            line = self.format_global_fold % (sale_date,
                                              training_days,
                                              model_name,
                                              scope,
                                              fold_number,
                                              max_depth,
                                              median_abs_error,
                                              median_rel_abs_error)
            yield line

    def zip_fold_lines(self, key, result):
        fold_number, sale_date, training_days, model_name, scope = key
        assert isinstance(scope, tuple)
        assert scope[0] == 'zip'
        zip_code = scope[1]
        for result_key, result_value in result.iteritems():
            assert result_key[0] == 'n_trees'
            n_trees = result_key[1]
            median_abs_error, median_rel_abs_error = errors(result_value)
            line = self.format_global_fold % (sale_date,
                                              training_days,
                                              model_name,
                                              zip_code,
                                              fold_number,
                                              n_trees,
                                              median_abs_error,
                                              median_rel_abs_error)
            yield line

    def summarize_global(self, sale_date, training_days, model_name, all_results, control):
        scope = 'global'
        median_errors = np.zeros(control.n_folds, dtype=np.float64)
        median_rel_errors = np.zeros(control.n_folds, dtype=np.float64)
        for fold_number in xrange(control.n_folds):
            key = (fold_number, sale_date, training_days, model_name, scope)
            if key not in all_results:
                # can happen when a model could not be fit
                print 'model_result missing key', key
                continue
            model_result = all_results[key]
            if len(model_result['actuals']) == 0:
                continue
            median_abs_error, median_rel_abs_error = errors(model_result)
            fold_line = self.format_global_fold % (sale_date,
                                                   training_days,
                                                   model_name,
                                                   scope,
                                                   fold_number,
                                                   median_abs_error,
                                                   median_rel_abs_error)
            print fold_line
            median_errors[fold_number] = median_abs_error
            median_rel_errors[fold_number] = median_rel_abs_error
        all_folds_line = self.format_global % (sale_date,
                                               training_days,
                                               model_name,
                                               scope,
                                               np.median(median_errors),
                                               np.median(median_rel_errors))
        print all_folds_line

    def summarize_zip(self, sale_date, training_days, model_name, all_results, control):

        def list_median(lst):
            assert(len(lst) > 0)
            return np.median(np.array(lst, dtype=np.float64))

        def report_zip_code(zip_code, keys):
            median_abs_errors = []
            median_rel_abs_errors = []
            for key in keys:
                model_result = all_results[key]
                median_abs_error, median_rel_abs_error = errors(model_result)
                fold_line = self.format_zip_fold % (sale_date,
                                                    training_days,
                                                    model_name,
                                                    zip_code,
                                                    key[0],  # fold number
                                                    median_abs_error,
                                                    median_rel_abs_error)
                print fold_line
                median_abs_errors.append(median_abs_error)
                median_rel_abs_errors.append(median_rel_abs_error)
            all_folds_line = self.format_zip % (sale_date,
                                                training_days,
                                                model_name,
                                                zip_code,
                                                list_median(median_abs_errors),
                                                list_median(median_rel_abs_errors))
            print all_folds_line

        # determine all zip codes in the specified lines
        zip_codes = collections.defaultdict(set)
        for key in all_results.keys():
            key_fold_number, key_sale_date, key_training_days, key_model_name, key_scope = key
            if key_scope == 'global':
                # examine only zip code scopes
                continue
            if key_sale_date == sale_date and key_training_days == training_days and key_model_name == model_name:
                key_zip_code = key_scope[1]
                zip_codes[key_zip_code].add(key)

        # process each zip code
        for zip_code, keys in zip_codes.iteritems():
            report_zip_code(zip_code, keys)

    def summarize(self, sale_date, training_days, model_name, all_results, control):
        self.summarize_global(sale_date, training_days, model_name, all_results, control)
        self.summarize_zip(sale_date, training_days, model_name, all_results, control)


class RfOLD(object):
    'Random forests via sklearn'
    def __init__(self):
        self.Model_Constructor = ensemble.RandomForestRegressor

    def reporter(self):
        return ReportRf

    def run(self, df_train, df_test, df_next, control):
        '''fit on train, test on test, return dict of variants

        The variants are defined by the number of trees in the forest

        RETURN dict with key = variant_description
        '''
        verbose = False

        def variant(max_depth):  # class Rf
            'per Andreas Mueller, regularize using max depth of each random tree'
            train_x = x(None, df_train, control)  # no transformation
            test_x = x(None, df_test, control)
            train_y = y(None, df_train, control)
            model = self.Model_Constructor(max_depth=max_depth,
                                           n_estimators=control.rf_n_estimators,
                                           random_state=control.random_seed)
            fitted_model = model.fit(train_x, train_y)
            estimates = fitted_model.predict(test_x)
            # return selected fitted results
            result = {
                'feature_importances': fitted_model.feature_importances_,
                'estimates': estimates,
                'actuals': y('None', df_test, control),
                'estimates_next': fitted_model.predict(x(None, df_next, control)),
                'actuals_next': y('None', df_next, control),
                'n_train': len(train_x),
            }
            if verbose:
                for k, v in result.iteritems():
                    print k, v
            return result

        all_variants = {}
        for max_depth in control.rf_max_depths:
            variant_value = variant(max_depth)
            key = ('max_depth', max_depth)
            all_variants[key] = variant_value
        return all_variants


class Rf(object):
    'Random forests regressor via sklearn'
    def __init__(self, n_estimates, max_depth, random_state, variants):
        self.model = ensemble.RandomForestRegressor(n_estimates=n_estimates,
                                                  max_depth=max_depth,
                                                    random_state=random_state)

    def fit(x_df, extract_x, y_df, extract_y, sample_weight):
        self.fit(extract_x(x_df), extract_y(y_df), sample_weight)

    def predict(x_df, extract_x):
        return predict(extract_x(x_df))

    def attributes():
        return {'estimators_': self.model.estimators_,
                'feature_importances_': self.model.feature_importances_,
                'oob_score_': self.model.oob_score_,
                'oob_prediction_': self.model.oob_precition_,
                }

    # OLD BELOW ME
    def reporter(self):
        return ReportRf

    def run(self, df_train, df_test, df_next, control):
        '''fit on train, test on test, return dict of variants

        The variants are defined by the number of trees in the forest

        RETURN dict with key = variant_description
        '''
        verbose = False

        def variant(max_depth):  # class Rf
            'per Andreas Mueller, regularize using max depth of each random tree'
            train_x = x(None, df_train, control)  # no transformation
            test_x = x(None, df_test, control)
            train_y = y(None, df_train, control)
            model = self.Model_Constructor(max_depth=max_depth,
                                           n_estimators=control.rf_n_estimators,
                                           random_state=control.random_seed)
            fitted_model = model.fit(train_x, train_y)
            estimates = fitted_model.predict(test_x)
            # return selected fitted results
            result = {
                'feature_importances': fitted_model.feature_importances_,
                'estimates': estimates,
                'actuals': y('None', df_test, control),
                'estimates_next': fitted_model.predict(x(None, df_next, control)),
                'actuals_next': y('None', df_next, control),
                'n_train': len(train_x),
            }
            if verbose:
                for k, v in result.iteritems():
                    print k, v
            return result

        all_variants = {}
        for max_depth in control.rf_max_depths:
            variant_value = variant(max_depth)
            key = ('max_depth', max_depth)
            all_variants[key] = variant_value
        return all_variants


class Rfnw(object):
    'random forest without weighted training samples'
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators

    def fit_and_predict(self, hp, train_df, validate_df, lhs_name, rhs_names):
        'fit one model and use it for all the predictions'
        pdb.set_trace()
        assert hp[0] == 'max_depth'
        max_depth = hp[1]
        m = Rf(max_depth=max_depth,
               n_estimatores=self.n_estimators,
               random_state=self.random_state)
        m.fit(x(None, train_df, rhs_names), y(None, train_df, lhs_name))
        estimates = m.predict(x(None, validate_df, rhs_names))
        actuals = y(None, validate_df, lhs_name)
        return {'estimates': estimates,
                'actuals': actuals,
                'hp': hp,
                'attributes': m.attributes(),
                }


class Rfw(object):
    'random forest with weighted training samples'
    def __init__(self, make_weights, n_estimators, max_depth, random_state):
        self.make_weights = make_weights

    def fit_and_predict(self, hp, train_df, validate_df):
        'fit one model for each prediction'
        pdb.set_trace()
        assert hp[0] == 'max_depth'
        max_depth = hp[1]
        results = []
        pdb.set_trace()
        for validate_row_df in validate_df.iterrows():
            m.Rf(max_depth, self.control)
            m.fit(x(None, train_df, self.control),
                  y(None, train_df, self.control),
                  make_weights(validate_row_df, train_df))
            estimates = m.predict(x(None, validate_row_df, self.control))
            actuals = y(None, validate_row_df, self.control)
            result = {'estimates': estimates,
                      'actuals': actuals,
                      'hp': hp,
                      'attributes': m.attributes(),
                      }
            results.append(result)
        pdb.set_trace()
        return results


class RfwOLD(object):
    'Random forests via sklearn'
    def __init__(self):
        self.Model_Constructor = ensemble.RandomForestRegressor

    def reporter(self):
        return ReportRf

    def run(self, df_train, df_test, df_next, weights_fn, control):
        '''fit on train, test on test, return dict of variants

        The variants are defined by the number of trees in the forest

        RETURN dict with key = variant_description
        '''
        verbose = False

        def variant(max_depth):  # in class Rfw
            'fit model to each test sample, as the weights vary by test sample'
            model = self.Model_Constructor(max_depth=max_depth,
                                           n_estimators=control.rf_n_estimators,
                                           random_state=control.random_seed)
            train_x = x(None, df_train, control)
            train_y = y(None, df_train, control)

            def fit_and_test(test_df):
                'fit a model for each sample in the test dataframe'
                actuals = np.zeros(len(test_df))
                estimates = np.zeros(len(test_df))
                feature_importances = []
                for i in xrange(len(test_df)):
                    test_row = test_df[i:(i + 1)]  # a dataframe
                    fitted = model.fit(train_x, train_y, weights_fn(test_row))
                    test_x = x(None, test_row, control)
                    estimate = fitted.predict(test_x)
                    estimates[i] = estimate
                    feature_importances.append(fitted.feature_importances_)
                    actual = y(None, test_row, control)[0]
                    actuals[i] = actual
                return estimates, feature_importances, actuals

            test_estimates, test_importances, test_actuals = fit_and_test(df_test)
            pdb.set_trace()
            next_estimates, _, next_actuals = fit_and_test(df_next)
            result = {
                'feature_importances': test_importances,
                'estimates': test_estimates,
                'actuals': test_actuals,
                'estimates_next': next_estimates,
                'actuals_next': next_actuals,
                'n_train': len(df_train)
            }
            if verbose:
                for k, v in result.iteritems():
                    print k, v
            pdb.set_trace()
            return result

        pdb.set_trace()
        all_variants = {}
        for max_depth in control.rf_max_depths:
            variant_value = variant(max_depth)
            key = ('max_depth', max_depth)
            all_variants[key] = variant_value
        return all_variants


class Rfw1(Rfw):
    'Random forests with samples weighted according to scheme 1'
    def __init__(self):
        pass

    def reporter(self):
        return ReportRf

    @staticmethod
    def make_always_ones(train_df):
        def always_ones(df_row):
            return np.ones(len(train_df))
        return always_ones

    def run(self, df_train, df_test, df_next, control):
        '''fit on train, test on test, return dict of variants

        The variants are defined by the number of trees in the forest

        RETURN dict with key = variant_description
        '''
        pdb.set_trace()
        weights_fn = Rfw1.make_always_ones(df_train)
        rfw = Rfw()
        return rfw.run(df_train, df_test, df_next, weights_fn, control)

def within(sale_date, training_days, df):
    'return indices of samples up to training_days before the sale_date'
    assert(training_days > 0)
    # one training day means use samples on the sale date only
    first_ok_sale_date = sale_date - datetime.timedelta(training_days - 1)
    date_column = 'sale.python_date'
    after = df[date_column] >= first_ok_sale_date
    before = df[date_column] <= sale_date
    ok_indices = np.logical_and(after, before)
    return ok_indices


def mask_in_date_range(df, date_range):
    df_date = df['sale.python_date']
    return (df_date >= date_range.first) & (df_date <= date_range.last)


def samples_in_date_range(df, date_range):
    'return new df'
    return df[mask_in_date_range(df, date_range)]


def on_sale_date(sale_date, df):
    '''return indices of sample on the sale date'''
    date_column = 'sale.python_date'
    result = df[date_column] == sale_date
    return result


def add_age(df, sale_date):
    'Return new df with extra columns for age and effective age'
    column_names = df.columns.tolist()
    if 'age' in column_names:
        print column_names
        print 'age in column_names'
        pdb.set_trace()
    assert('age' not in column_names)
    assert('age2' not in column_names)
    assert('effective.age' not in column_names)
    assert('effective.age2' not in column_names)

    sale_year = df['sale.year']

    def age(column_name):
        'age from sale_date to specified column'
        age_in_years = sale_year - df[column_name].values
        return pd.Series(age_in_years, index=df.index)

    result = df.copy(deep=True)

    result['age'] = age('YEAR.BUILT')
    result['effective.age'] = age('EFFECTIVE.YEAR.BUILT')
    result['age2'] = result['age'] * result['age']
    result['effective.age2'] = result['effective.age'] * result['effective.age']

    return result


def unique_zip_codes(df):
    'yield each unique zip code in the dataframe'
    unique_zip_codes = df['zip5'].unique()
    for i in xrange(len(unique_zip_codes)):
        yield unique_zip_codes[i]


def zip_codes(df, a_zip_code):
    'return new dataframe containing just the specified zip code'
    df_copy = df.copy(deep=True)
    result = df_copy[df_copy['zip5'] == a_zip_code]
    return result


def make_train_modelOLD(df, sale_date, training_days):
    'return df of transactions no more than training_days before the sale_date'
    just_before_sale_date = within(sale_date, training_days, df)
    train_model = add_age(df[just_before_sale_date], sale_date)
    return train_model


def make_test_modelOLD(df, sale_date):
    'return df of transactions on the sale_date'
    selected_indices = on_sale_date(sale_date, df)
    test_model = add_age(df[selected_indices], sale_date)
    return test_model


def determine_most_popular_zip_code(df, control):
    'return the zip_code that occurs most ofen in the dataframe'
    zip_code_counter = collections.Counter()
    for _, zip_code in df.zip5.iteritems():
        zip_code_counter[zip_code] += 1
    most_common_zip_code, count = zip_code_counter.most_common(1)[0]
    print 'most common zip_code', most_common_zip_code, 'occurs', count

    # assert: the most common zip code is in each fold
    fold_number = -1
    folds_for_zip_code = collections.defaultdict(set)
    kf = cross_validation.KFold(n=(len(df)),
                                n_folds=control.n_folds,
                                shuffle=True,
                                random_state=control.random_seed)
    for train_indices, test_indices in kf:
        fold_number += 1
        train = df.iloc[train_indices].copy(deep=True)
        test = df.iloc[test_indices].copy(deep=True)
        if most_common_zip_code not in test.zip5.values:
            print most_common_zip_code, 'not in', fold_number
        for zip_code in unique_zip_codes(test):
            assert(zip_code in test.zip5.values)
            if zip_code not in train.zip5.values:
                print 'fold %d zip_code %d in test and not train' % (
                    fold_number,
                    zip_code)
            folds_for_zip_code[zip_code].add(fold_number)
    assert(len(folds_for_zip_code[most_common_zip_code]) == 10)

    # print zip_code not in each test set
    count_in_10 = 0
    count_not_in_10 = 0
    for zip_code, set_folds in folds_for_zip_code.iteritems():
        if len(set_folds) != 10:
            print 'zip_code %d in only %d folds' % (zip_code, len(set_folds))
            count_not_in_10 += 1
        else:
            count_in_10 += 1
    print 'all other zip codes are in 10 folds'
    print 'in 10: %d  not in 10: %d' % (count_in_10, count_not_in_10)
    print 'NOTE: all this analysis is before training samples are selected'



def read_training_data(control):
    'return dataframe'


class AccumulateMedianErrors():
    def __init__(self):
        self.dfa = DataframeAppender([('fold_number', np.int64),
                                      ('training_days', np.int64),
                                      ('model_id', object),  # string: model + hyperparameters
                                      ('scope', object),     # 'global' or zip code
                                      ('median_abs_error', np.float64),
                                      ('median_rel_error', np.float64),
                                      ('n_samples', np.float64)])

    def accumulate(self, key, result):
        verbose = False
        fold_number, sale_date, training_days, model_name, scope = key
        if model_name == 'rf':
            self._accumulate_rf(fold_number, training_days, scope, result)
        elif model_name == 'lr':
            self._accumulate_lr(fold_number, training_days, scope, result)
        else:
            raise RuntimeError('bad model_name: ' + str(model_name))
        if verbose:
            print self.dfa.df

    def _accumulate_lr(self, fold_number, training_days, scope, result):
        for k, v in result.iteritems():
            model_id = 'lr ' + str(k[1])[:3] + ' ' + str(k[3])[:3]
            self._append(fold_number, training_days, model_id, scope, v)

    def _accumulate_rf(self, fold_number, training_days, scope, result):
        for k, v in result.iteritems():
            model_id = 'rf ' + str(k[1])
            self._append(fold_number, training_days, model_id, scope, v)

    def _append(self, fold_number, training_days, model_id, scope, model_result):
        median_abs_error, median_rel_error = errors(model_result)
        self.dfa.append([fold_number,
                         training_days,
                         model_id,
                         scope if scope == 'global' else str(scope[1]),
                         median_abs_error,
                         median_rel_error,
                         len(model_result['actuals'])])

    def dataframe(self):
        return self.dfa.result()


def squeeze(result, verbose=False):
    'replace float64 with float32'

    def is_np_array_float64(x):
        return isinstance(x, np.ndarray) and x.dtype == np.float64

    def is_np_scalar_float64(x):
        return isinstance(x, np.float64)

    def squeeze_np(value):
        if is_np_array_float64(value):
            return np.array(value, dtype=np.float32)
        elif is_np_scalar_float64(value):
            return np.float32(value)
        elif isinstance(value, (float, int, long, complex)):
            return value  # don't convert scalar numbers
        else:
            print value
            raise RuntimeError('unexpected')

    if verbose:
        pprint(result)
    assert(isinstance(result, dict))
    new_result = {}
    for k, v in result.iteritems():
        if isinstance(k, tuple):
            # ols result
            new_v = {key: squeeze_np(value) for key, value in v.iteritems()}
            new_result[k] = new_v
        else:
            # unexpected
            print k, v
            raise RuntimeError('unexpected')

    if verbose:
        pprint(new_result)
    return new_result


def make_weights(query, train_df, hpw, control):
    'return numpy.array of weights for each sample'
    if hpw == 1:
        return np.ones(len(train_df))
    else:
        print 'bad hpw: %s' % hpw


CvKey = collections.namedtuple('CvKey', 'model hp1 hp2 td fn')
CvValue = collections.namedtuple('CvValue', 'actuals estimates attributes')


def sweep_hp_lr(train_df, validate_df, control):
    'sweep hyperparameters, fitting and predicting for each combination'
    def x_matrix(df, transform):
        augmented = add_age(df, control.arg.sale_date)
        return x(transform, augmented, control.predictors)

    def y_vector(df, transform):
        return y(transform, df, control.price_column)

    verbose = True
    LR = linear_model.LinearRegression
    results = {}
    for hpx in control.arg.hpx:
        for hpy in control.arg.hpy:
            if verbose:
                print 'sweep_hr_lr hpx %s hpy %s' % (hpx, hpy)
            model = LR(fit_intercept=True,
                       normalize=True,
                       copy_X=False,
                       )
            train_x = x_matrix(train_df, hpx)
            train_y = y_vector(train_df, hpy)
            model.fit(train_x, train_y)
            estimates = model.predict(x_matrix(validate_df, hpx))
            actuals = y_vector(validate_df, hpy)
            attributes = {
                'coef_': model.coef_,
                'intercept_': model.intercept_
            }
            results[('y_transform', hpy), ('x_transform', hpx)] = {
                'estimate': estimates,
                'actual': actuals,
                'attributes': attributes
            }
    return results


def sweep_hp_rf(train_df, validate_df, control):
    'fit a model and validate a model for each hyperparameter'
    def x_matrix(df):
        augmented = add_age(df, control.arg.sale_date)
        return x(None, augmented, control.predictors)

    def y_vector(df):
        return y(None, df, control.price_column)

    verbose = True
    RFR = ensemble.RandomForestRegressor
    train_x = x_matrix(train_df)
    train_y = y_vector(train_df)
    results = {}
    for hpd in control.arg.hpd:
        for hpw in control.arg.hpw:
            for validate_row_index in xrange(len(validate_df)):
                if verbose:
                    print 'sweep_hp_rf hpd %d hpw %d validate_row_index %d of %d' % (
                        hpd, hpw, validate_row_index, len(validate_df))
                validate_row = validate_df[validate_row_index: validate_row_index + 1]
                model = RFR(n_estimators=control.n_rf_estimators,  # number of trees
                            random_state=control.random_seed,
                            max_depth=hpd)
                weights = make_weights(validate_row, train_df, hpw, control)
                model.fit(train_x, train_y, weights)
                estimate = model.predict(x_matrix(validate_row))[0]
                actual = y_vector(validate_row)[0]
                # Don't keep some attributes
                #  oob attributes are not produced because we didn't ask for them
                #  estimators_ contains a fitted model for each estimate
                attributes = {
                    'feature_importances_': model.feature_importances_,
                }
                results[('max_depth', hpd), ('weight_scheme_index', hpw)] = {
                    'estimate': estimate,
                    'actual': actual,
                    'attributes': attributes,
                }
    return results


def cross_validate(df, control):
    'produce estimated generalization errors'
    verbose = True
    results = {}
    fold_number = -1
    sale_dates_mask = mask_in_date_range(df, control.sale_date_range)
    skf = cross_validation.StratifiedKFold(sale_dates_mask, control.n_folds)
    for train_indices, validate_indices in skf:
        fold_number += 1
        fold_train_all = df.iloc[train_indices].copy(deep=True)
        fold_validate_all = df.iloc[validate_indices].copy(deep=True)
        for td in control.arg.td:
            if verbose:
                print 'cross_validate fold %d of %d training_days %d' % (
                    fold_number, control.n_folds, td)
            fold_train = samples_in_date_range(
                fold_train_all,
                DateRange(first=control.arg.sale_date - datetime.timedelta(td),
                          last=control.arg.sale_date - datetime.timedelta(1))
            )
            fold_validate = samples_in_date_range(
                fold_validate_all,
                control.sale_date_range
            )
            if control.arg.model == 'lr':
                d = sweep_hp_lr(fold_train, fold_validate, control)
            elif control.arg.model == 'rf':
                d = sweep_hp_rf(fold_train, fold_validate, control)
                # d = cross_validate_rf(fold_train, fold_validate, control)
            else:
                print 'bad model: %s' % control.model
                pdb.set_trace()
            results[(('fn', fold_number), ('td', td))] = d
    return results


def predict_next(df, control):
    'fit each model and predict transaction in next period'
    verbose = True
    for td in control.arg.td:
        if verbose:
            print 'predict_next training_days %d' % td
        last_sale_date = control.sale_date_range.last
        train_df = samples_in_date_range(
            df,
            DateRange(first=last_sale_date - datetime.timedelta(td),
                      last=last_sale_date)
        )
        next_days = 30 if control.arg.month else 7
        test_df = samples_in_date_range(
            df,
            DateRange(first=last_sale_date,
                      last=last_sale_date + datetime.timedelta(next_days))
        )
        if control.arg.model == 'lr':
            return sweep_hp_lr(train_df, test_df, control)
        elif control.arg.model == 'rf':
            return sweep_hp_rf(train_df, test_df, control)
        else:
            print 'bad model: %s' % control.arg.model


def fit_and_test_models(df_all, control):
    'Return all_results dict'
    # throw away irrelevant transactions
    df_relevant = samples_in_date_range(df_all, control.relevant_date_range)

    results_cv = cross_validate(df_relevant, control)
    results_next = predict_next(df_relevant, control)

    pdb.set_trace()
    return results_cv, results_next


def print_results(all_results, control):
    for training_days in control.training_days:
        for model_name, model in control.models.iteritems():
            report = model.reporter()()  # how to print is in the model result
            report.summarize(control.sale_date,
                             training_days,
                             model_name,
                             all_results,
                             control)


def write_all_results(all_results, control):
    for k, v in all_results.iteritems():
        k_fold_number, k_sale_date, k_training_days, k_model_name, k_scope = k
        assert k_scope == 'global', k_scope  # code doesn't work for scope == 'zip'
        for variant, result in v.iteritems():
            # possible create directory WORKING/YYYY-MM-DD/MODEL-TD/
            dir_path = '%s%s-%03d/' % (control.dir_out, k_model_name, k_training_days)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            hp = (
                '%02d' % variant[1] if k_model_name == 'rf' else
                '%3s-%3s' % (variant[1][:3], variant[3][:3]) if k_model_name == 'lr' else
                None
            )
            file_path = '%s%s-%d.pickle' % (dir_path, hp, k_fold_number)
            print 'about to write', file_path
            f = open(file_path, 'wb')
            pickle.dump({'control': control, 'variant': variant, 'result': result}, f)
            f.close()


def main(argv):
    #  warnings.filterwarnings('error')  # convert warnings to errors
    control = make_control(argv)

    sys.stdout = Logger(logfile_path=control.path_log)  # print also write to log file
    print control

    # read input
    if control.use_old_input:
        f = open(control.path_in_old, 'rb')
        df_loaded = pickle.load(f)
        f.close()
    else:
        df_loaded = pd.read_csv(control.path_in, engine='c')

    df_loaded_copy = df_loaded.copy(deep=True)  # make sure df_loaded isn't changed
    if False:
        most_popular_zip_code = determine_most_popular_zip_code(df_loaded.copy(), control)
        print most_popular_zip_code

    results_cv, results_next = fit_and_test_models(df_loaded, control)
    assert(df_loaded.equals(df_loaded_copy))

    # write results
    def file_name(key):
        'model-foldNumber-trainingDays'
        assert len(key) == 2, key
        s = '%s-%s-%s' % (control.arg.model, key[0], key[1])
        return s

    def write(dir_prefix, results):
        for k, v in results.iteritems():
            directory = control.dir_out + dir_prefix
            if not os.path.exists(directory):
                os.makedirs(directory)
            f = open(directory + file_name(k), 'wb')
            pickle.dump((k, v), f)
            f.close()

    write('cv/', results_cv)
    write('next/', results_next)

    print 'ok'


if __name__ == "__main__":
    if False:
        # quite pyflakes warnings
        pdb.set_trace()
        pprint(None)
        np.all()
        pd.Series()
    main(sys.argv)
