'''create files contains estimated generalization errors for model

sys.argv: year month day
    The sale_date of the models. Uses data up to the day before the sale date

Files created:
    year-month-day-MODEL-SCOPE-T[-MODELPARAMS].foldResult
where:
    MODEL is one of the models {ols, lasso, ridge, rf, xt}
    SCOPE is one of {global,zDDDDD} (global or zip-code specific)
    T is the number of days in the training period
    MODELPARAMS depends on the model and may be empty
'''

import collections
import cPickle as pickle
import datetime
import numpy as np
import pandas as pd
import pdb
from pprint import pprint
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import ensemble
import sys
import warnings

from Bunch import Bunch
from directory import directory
from Logger import Logger


def usage():
    print 'usage: python ege-date.py yyyy-mm-dd'
    sys.exit(1)


def make_control(argv):
    'Return control Bunch'''
    script_name = argv[0]
    base_name = script_name.split('.')[0]
    random_seed = 123
    now = datetime.datetime.now()
    log_file_name = base_name + '.' + now.isoformat('T') + '.log'

    if len(argv) < 2:
        print 'missing date argument'
        usage()

    if len(argv) > 2:
        print 'extra args'
        usage()

    year, month, day = argv[1].split('-')
    sale_date = datetime.date(int(year), int(month), int(day))

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

    debug = False
    b = Bunch(
        path_in=directory('working') + 'transactions-subset2.pickle',
        path_log=directory('log') + log_file_name,
        path_out=directory('working') + base_name + '-%s' % sale_date + '.pickle',
        arg_date=sale_date,
        random_seed=random_seed,
        sale_date=sale_date,
        models={'rf': Rf(), 'ols': Ols()} if not debug else {'rf': Rf()},
        scopes=['global', 'zip'],
        training_days=(7, 366) if debug else range(7, 366, 7),
        n_folds=10,
        predictors=predictors,
        price_column='SALE.AMOUNT',
        debug=debug)
    return b


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
    pprint(model_result)
    actuals = model_result['actuals']
    estimates = model_result['estimates']
    abs_error = np.abs(actuals - estimates)
    median_abs_error = np.median(abs_error)
    rel_abs_error = abs_error / actuals
    median_rel_abs_error = np.median(rel_abs_error)
    return median_abs_error, median_rel_abs_error


class ReportOls(object):
    'report generation with y_mode and x_mode in key'
    # NOTE: perhaps reusable for any model with y and x modes

    def __init__(self):
        self.format_global_fold = '%10s %2d %3s %6s %3s %3s f%d %6.0f %3.2f'
        self.format_zip_fold = '%10s %2d %3s %6d %3s %3s f%d %6.0f %3.2f'
        self.format_global = '%10s %2d %3s %6s %3s %3s median %6.0f %3.2f'
        self.format_zip = '%10s %2d %3s %6d %3s %3s median %6.0f %3.2f'

    def global_fold_line(self, key, result):
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
            return line

    def zip_fold_line(self, key, result):
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
            return line

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


class Ols(object):
    'Ordinary least squares via sklearn'
    def __init__(self):
        self.Model_Constructor = linear_model.LinearRegression

    def reporter(self):
        return ReportOls

    def run(self, train, test, control):
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
            train_x = x(x_mode, train, control)
            test_x = x(x_mode, test, control)
            train_y = y(y_mode, train, control)
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
                'actuals': y('linear', test, control)
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
        self.format_global_fold = '%10s %2d %3s %6s f%d %6.0f %3.2f'
        self.format_zip_fold = '%10s %2d %3s %6d f%d %6.0f %3.2f'
        self.format_global = '%10s %2d %3s %6s  median %6.0f %3.2f'
        self.format_zip = '%10s %2d %3s %6d  median %6.0f %3.2f'

    def global_fold_line(self, key, result):
        fold_number, sale_date, training_days, model_name, scope = key
        assert(scope == 'global')
        median_abs_error, median_rel_abs_error = errors(result)
        line = self.format_global_fold % (sale_date,
                                          training_days,
                                          model_name,
                                          scope,
                                          fold_number,
                                          median_abs_error,
                                          median_rel_abs_error)
        return line

    def zip_fold_line(self, key, result):
        fold_number, sale_date, training_days, model_name, scope = key
        assert(isinstance(scope, tuple))
        assert(scope[0] == 'zip')
        zip_code = scope[1]
        median_abs_error, median_rel_abs_error = errors(result)
        line = self.format_global_fold % (sale_date,
                                          training_days,
                                          model_name,
                                          zip_code,
                                          fold_number,
                                          median_abs_error,
                                          median_rel_abs_error)
        return line

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

    def run(self, train, test, control):
        '''fit on train, test on test, return dict of variants

        The variants are defined by the number of trees in the forest

        RETURN dict with key = variant_description
        '''
        verbose = False

        def variant(n_trees):
            train_x = x(None, train, control)  # no transformation
            test_x = x(None, test, control)
            train_y = y(None, train, control)
            model = self.Model_Constructor(n_estimators=n_trees,
                                           random_state=control.random_seed)
            fitted_model = model.fit(train_x, train_y)
            estimates = fitted_model.predict(test_x)
            # return selected fitted results
            result = {
                'feature_importances': fitted_model.feature_importances_,
                'estimates': estimates,
                'actuals': y('None', test, control)}
            if verbose:
                for k, v in result.iteritems():
                    print k, v
            return result

        all_variants = {}
        for n_trees in (10, 100, 300, 1000):
            variant_value = variant(n_trees)
            key = ('n_trees', n_trees)
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


def squeeze(result, verbose=False):
    'replace float64 with float32'

    def is_np_array_float64(x):
        return isinstance(x, np.ndarray) and x.dtype == np.float64

    def is_np_scalar_float64(x):
        return isinstance(x, np.float64)

    if verbose:
        pprint(result)
    assert(isinstance(result, dict))
    new_result = {}
    for k, v in result.iteritems():
        if isinstance(k, str):
            # rf result
            if is_np_array_float64(v):
                # e.g., actual, estimate, other info in a vector
                new_result[k] = np.array(v, dtype=np.float32)
            else:
                print k, v
                raise RuntimeError('unexpected')
        elif isinstance(k, tuple):
            # ols result
            new_ols_result = {}
            for ols_key, ols_value in v.iteritems():
                if is_np_array_float64(ols_value):
                    new_ols_result[ols_key] = np.array(ols_value, dtype=np.float32)
                elif is_np_scalar_float64(ols_value):
                    new_ols_result[ols_key] = np.float32(ols_value)
                else:
                    print ols_key, ols_value
                    raise RuntimeError('unexpected')
            new_result[k] = new_ols_result
        else:
            # unexpected
            print k, v
            raise RuntimeError('unexpected')

    if verbose:
        pprint(new_result)
    return new_result


def fit_and_test_models(df, control):
    '''Return all_results dict and list of feature names

    all_results has results for each fold, sale date, training period model, scope
    '''
    verbose = False
    all_results = {}
    fold_number = -1
    on_sale_date = df['sale.python_date'] == control.sale_date
    num_test_samples = sum(on_sale_date)
    print 'sale_date %s has %d samples' % (control.sale_date, num_test_samples)
    assert num_test_samples >= control.n_folds, 'unable to form folds'
    skf = cross_validation.StratifiedKFold(on_sale_date, control.n_folds)
    for train_indices, test_indices in skf:
        fold_number += 1
        # don't create views (just to be careful)
        train = df.iloc[train_indices].copy(deep=True)
        test = df.iloc[test_indices].copy(deep=True)
        for training_days in control.training_days:
            train_model = make_train_model(train, control.sale_date, training_days)
            if len(train_model) == 0:
                print 'no training data fold %d sale_date %s training_days %d' % (
                    fold_number, control.sale_date, training_days)
                sys.exit(1)
            test_model = make_test_model(test, control.sale_date)
            if len(test_model) == 0:
                print 'no testing data fold %d sale_date %s training_days %d' % (
                    fold_number, control.sale_date, training_days)
                continue
            for model_name, model in control.models.iteritems():
                print fold_number, control.sale_date, training_days, model_name

                def make_key(scope):
                    return (fold_number, control.sale_date, training_days, model_name, scope)

                # determine global results (for all areas)
                if len(test_model) == 0 or len(train_model) == 0:
                    print 'skipping global zero length', len(test_model), len(train_model)
                else:
                    global_result = model.run(train=train_model,
                                              test=test_model,
                                              control=control)
                    key = make_key(scope='global')
                    all_results[key] = squeeze(global_result)
                    report = model.reporter()()  # instantiate report class
                    if verbose:
                        print report.global_fold_line(key, global_result)

                # determine results for each zip code in test data
                for zip_code in unique_zip_codes(test_model):
                    train_model_zip = zip_codes(train_model, zip_code)
                    test_model_zip = zip_codes(test_model, zip_code)
                    if len(train_model_zip) == 0 or len(test_model_zip) == 0:
                        print 'skipping zip zero length', zip_code, len(test_model_zip), len(train_model_zip)
                    else:
                        zip_code_result = model.run(train=train_model_zip,
                                                    test=test_model_zip,
                                                    control=control)
                        key = make_key(scope=('zip', zip_code))
                        all_results[key] = squeeze(zip_code_result)
                        if verbose:
                            print report.zip_fold_line(key, zip_code_result)
    print 'num test samples across all folds:', num_test_samples
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


def main(argv):
    warnings.filterwarnings('error')  # convert warnings to errors
    control = make_control(argv)

    sys.stdout = Logger(logfile_path=control.path_log)  # print also write to log file
    print control

    # read input
    f = open(control.path_in, 'rb')
    df_loaded = pickle.load(f)
    f.close()

    df_loaded_copy = df_loaded.copy(deep=True)  # used for debugging
    if False:
        most_popular_zip_code = determine_most_popular_zip_code(df_loaded.copy(), control)
        print most_popular_zip_code

    all_results = fit_and_test_models(df_loaded, control)
    assert(df_loaded.equals(df_loaded_copy))

    if False:
        print_results(all_results, control)

    # write result
    print 'writing results to', control.path_out
    result = {'control': control,  # control.predictors orders the x values
              'all_results': all_results}
    f = open(control.path_out, 'wb')
    pickle.dump(result, f)

    print 'ok'


if __name__ == "__main__":
    if False:
        # quite pyflakes warnings
        pdb.set_trace()
        pprint(None)
        np.all()
        pd.Series()
    main(sys.argv)
