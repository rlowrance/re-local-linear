'''create files contains estimated generalization errors for model

INPUT FILE
 WORKING/transactions-subset2.pickle

OUTPUT FILES
 WORKING/ege_week/YYYY-MM-DD/MODEL-TD/HP-FOLD.pickle  dict all_results
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
    print 'usage: python ege_week.py YYYY-MM-DD [--test] [--global]'
    print ' YYYY-MM-DD       mid-point of week; analyze -3 to +3 days'
    print ' --zip            optional; create zip-based sample as well as global'
    print ' --model {lr|rf}  which model to run'
    print ' --td start [stop [step]]  training_days'
    print ' --hp start [stop [step]]  required iff model is rf; hyperparameters to model'
    sys.exit(1)


def make_control(argv):
    'Return control Bunch'''

    print 'argv'
    pprint(argv)

    if len(argv) < 3:
        usage('missing parameters')

    script_name = argv[0]

    base_name = script_name.split('.')[0]
    random_seed = 123
    now = datetime.datetime.now()

    # prior work found that the assessment was not useful
    # just the census and tax roll features
    # predictors with transformation to log domain
    predictors = {  # the columns in the x_arrays are in this order
        'fraction.owner.occupied': None,
        'FIREPLACE.NUMBER': 'log1p',
        'BEDROOMS': 'log1p',
        'BASEMENT.SQUARE.FEET': 'log1p',
        'LAND.SQUARE.FOOTAGE': 'log',
        'zip5.has.industry': None,
        'census.tract.has.industry': None,
        'census.tract.has.park': None,
        'STORIES.NUMBER': 'log1p',
        'census.tract.has.school': None,
        'TOTAL.BATHS.CALCULATED': 'log1p',
        'median.household.income': 'log',  # not log feature in earlier version
        'LIVING.SQUARE.FEET': 'log',
        'has.pool': None,
        'zip5.has.retail': None,
        'census.tract.has.retail': None,
        'is.new.construction': None,
        'avg.commute': None,
        'zip5.has.park': None,
        'PARKING.SPACES': 'log1p',
        'zip5.has.school': None,
        'TOTAL.ROOMS': 'log1p',
        'age': None,
        'age2': None,
        'effective.age': None,
        'effective.age2': None}

    print 'number of predictors', len(predictors)

    # option YYYY-MM-DD (not optional)
    year, month, day = argv[1].split('-')
    sale_date = datetime.date(int(year), int(month), int(day))

    # option --model
    model = parse_command_line.get_arg(argv, '--model')
    if model == 'lr':
        models = {'lr': Lr()}
    elif model == 'rf':
        models = {'rf': Rf()}
    else:
        usage('unknown model')

    # option --zip
    include_zip = parse_command_line.has_arg(argv, '--zip')

    # option --td and --hp
    def make_range(tag):
        value = parse_command_line.get_arg(argv, tag)
        if value is None:
            usage('missing --hp')
        if isinstance(value, str):
            return (int(value),)
        if isinstance(value, list):
            if len(value) == 1:
                return (int(value[0]),)
            if len(value) == 2:
                return range(int(value[0], value[1]))
            if len(value) == 3:
                return range(int(value[0]), int(value[1]), int(value[2]))
        usage('incorrect values for ' + tag)

    td_range = make_range('--td')
    hp_range = make_range('--hp') if parse_command_line.has_arg(argv, '--hp') else None
    if model == 'lr':
        if hp_range is not None:
            usage('do not provide --hp for lr models')
    else:
        if hp_range is None:
            usage('must provide --hp for lr models')

    log_file_name = base_name + '.' + now.isoformat('T') + '.log'

    if model == 'lr' and hp_range is not None:
        usage('do not supply hyperparameters for lr models')

    # FIXME: the out file name will vary over the range of hp and td values

    debug = False
    test = False

    b = Bunch(
        path_in=directory('working') + 'transactions-subset2.pickle',
        path_log=directory('log') + log_file_name,
        dir_out=directory('working') + base_name + '/' + argv[1] + '/',
        start_time=now,
        random_seed=random_seed,
        sale_date=sale_date,
        base_name=base_name,
        models=models,
        scopes=('global', 'zip') if include_zip else ('global',),
        training_days=td_range,
        rf_max_depths=hp_range,
        rf_n_estimators=1000,  # number of trees in each random forest
        n_folds=10,
        predictors=predictors,
        price_column='SALE.AMOUNT',
        test=test,
        debug=debug)
    return b


def elapsed_time(start_time):
    return datetime.datetime.now() - start_time


def x(mode, df, control):
    '''return 2D np.array, with df x values possibly transformed to log

    RETURNS array: np.array 2D
    '''
    def transform(v, mode, transformation):
        if mode is None:
            return v
        if mode == 'linear':
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

    array = np.empty(shape=(df.shape[0], len(control.predictors)),
                     dtype=np.float64).T
    # build up in transposed form
    index = 0
    for predictor_name, transformation in control.predictors.iteritems():
        v = transform(df[predictor_name].values, mode, transformation)
        array[index] = v
        index += 1
    return array.T


def y(mode, df, control):
    '''return np.array 1D with transformed price column from df'''
    df2 = df.copy(deep=True)
    if mode == 'log':
        df2[control.price_column] = \
            pd.Series(np.log(df[control.price_column]),
                      index=df.index)
    array = np.array(df2[control.price_column].as_matrix(), np.float64)
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


class Rf(object):
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

        def variant(max_depth):
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


def is_between(df, first_date, last_date):
    'return mask for df containing subset of samples between the two dates'
    df_date = df['sale.python_date']
    return (df_date >= first_date) & (df_date <= last_date)


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


def make_train_model(df, sale_date, training_days):
    'return df of transactions no more than training_days before the sale_date'
    just_before_sale_date = within(sale_date, training_days, df)
    train_model = add_age(df[just_before_sale_date], sale_date)
    return train_model


def make_test_model(df, sale_date):
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

    return most_common_zip_code


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


def fit_and_test_models(df_all, control):
    'Return all_results dict'
    verbose = False

    # determine samples that are in the test period ( = 1 week around the sale_date)
    first_sale_date = control.sale_date - datetime.timedelta(3)
    last_sale_date = control.sale_date + datetime.timedelta(3)
    in_sale_period = is_between(df=df_all,
                                first_date=first_sale_date,
                                last_date=last_sale_date)
    num_sale_samples = sum(in_sale_period)
    print 'num sale samples', num_sale_samples
    assert num_sale_samples >= control.n_folds, 'unable to form folds'

    # test data is the next week after the last training sample
    df_next = add_age(df_all[is_between(df_all,
                                        last_sale_date + datetime.timedelta(1),
                                        last_sale_date + datetime.timedelta(7))],
                      sale_date=last_sale_date + datetime.timedelta(7))

    print 'df_next has %d samples' % len(df_next)

    all_results = {}
    fold_number = -1
    skf = cross_validation.StratifiedKFold(in_sale_period, control.n_folds)
    for train_indices, test_indices in skf:
        fold_number += 1
        # don't create views (just to be careful)
        df_train = df_all.iloc[train_indices].copy(deep=True)
        df_test = df_all.iloc[test_indices].copy(deep=True)
        for training_days in control.training_days:
            assert training_days > 0
            # determine training samples for the models
            df_train_model = \
                add_age(df_train[is_between(df=df_train,
                                            first_date=first_sale_date - datetime.timedelta(training_days),
                                            last_date=first_sale_date - datetime.timedelta(1))],
                        first_sale_date)
            if len(df_train_model) == 0:
                print 'no training data fold %d training_days %d' % (
                    fold_number, training_days)
                sys.exit(1)

            # determine testing samples for the models
            df_test_model = \
                add_age(df_test[is_between(df=df_test,
                                           first_date=first_sale_date,
                                           last_date=last_sale_date)],
                        first_sale_date)
            if len(df_test_model) == 0:
                print 'no testing data fold %d sale_date %s training_days %d' % (
                    fold_number, control.sale_date, training_days)
                continue

            print 'model samples sizes: training_days %d train %d test %d' % (
                training_days, len(df_train_model), len(df_test_model))

            # fit and test each model
            for model_name, model in control.models.iteritems():
                print '%d %s %d %s elapsed %s' % (
                    fold_number, control.sale_date, training_days, model_name,
                    elapsed_time(control.start_time))

                def make_key(scope):
                    return (fold_number, control.sale_date, training_days, model_name, scope)

                # determine global results (for all areas)
                if len(df_test_model) == 0 or len(df_train_model) == 0:
                    print 'skipping global zero length: #test %d #train %d' % (
                        len(df_test_model), len(df_train_model))
                    continue

                if 'global' in control.scopes:
                    global_result = model.run(df_train=df_train_model,
                                              df_test=df_test_model,
                                              df_next=df_next,
                                              control=control)
                    global_key = make_key(scope='global')
                    all_results[global_key] = squeeze(global_result)
                    if verbose:
                        report = model.reporter()()  # instantiate report class
                        for line in report.global_fold_lines(global_key, global_result):
                            print line

                # determine results for each zip code in test data
                if 'zip' in control.scopes:
                    for zip_code in unique_zip_codes(df_test_model):
                        df_train_model_zip = zip_codes(df_train_model, zip_code)
                        df_test_model_zip = zip_codes(df_test_model, zip_code)
                        if len(df_train_model_zip) == 0 or len(df_test_model_zip) == 0:
                            print 'skipping zip zero length: zip %d #test %d #train %d' % (
                                zip_code, len(df_test_model_zip), len(df_train_model_zip))
                        else:
                            zip_code_result = model.run(df_train=df_train_model_zip,
                                                        df_test=df_test_model_zip,
                                                        df_next=df_next,
                                                        control=control)
                            zip_code_key = make_key(scope=('zip', zip_code))
                            all_results[zip_code_key] = squeeze(zip_code_result)
                            if verbose:
                                for line in report.zip_fold_lines(zip_code_key, zip_code_result):
                                    print line
    print 'num sale samples across all folds:', num_sale_samples
    return all_results


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
    warnings.filterwarnings('error')  # convert warnings to errors
    control = make_control(argv)

    sys.stdout = Logger(logfile_path=control.path_log)  # print also write to log file
    print control

    # read input
    f = open(control.path_in, 'rb')
    df_loaded = pickle.load(f)
    f.close()

    df_loaded_copy = df_loaded.copy(deep=True)  # make sure df_loaded isn't changed
    if False:
        most_popular_zip_code = determine_most_popular_zip_code(df_loaded.copy(), control)
        print most_popular_zip_code

    all_results = fit_and_test_models(df_loaded, control)
    assert(df_loaded.equals(df_loaded_copy))

    if False:
        # this code doesn't know about the variants for the Rf model
        # in addition, we don't need these results, because downstream programs have
        # been written to summarize the results
        print_results(all_results, control)

    # write result
    write_all_results(all_results, control)

    print 'ok'


if __name__ == "__main__":
    if False:
        # quite pyflakes warnings
        pdb.set_trace()
        pprint(None)
        np.all()
        pd.Series()
    main(sys.argv)
