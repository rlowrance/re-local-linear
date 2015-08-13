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
import sys
import warnings

from Bunch import Bunch
from directory import directory
from Logger import Logger

# import FoldResult
# import ModelsOls


if False:
    # quiet warnings from pyflakes
    pdb.set_trace()
    pprint(None)
    np.all()
    pd.Series()


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
    predictors = {
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

    debug = True
    b = Bunch(
        path_in=directory('working') + 'transactions-subset2.pickle',
        path_log=directory('log') + log_file_name,
        arg_date=sale_date,
        random_seed=random_seed,
        sale_dates=[sale_date],
        models={'ols': Ols()},
        scopes=['global', 'zip'],
        training_days=range(7, 14 if debug else 366, 7),
        n_folds=10,
        predictors=predictors,
        price_column='SALE.AMOUNT',
        debug=debug)
    return b


def x(mode, df, control):
    '''return 2D np.array, with df x values possibly transformed to log

    RETURNS
    array: np.array 2D
    names: list of column names for array
    '''
    def transform(v, mode, transformation):
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
    return array.T, control.predictors.keys()


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


class Ols(object):
    'Ordinary least squares via sklearn'
    def __init__(self):
        self.Model_Constructor = linear_model.LinearRegression

    def fit_and_predictOLD(self, train_x, train_y, test_x):
        'return predictions and fitted model'
        if train_x.size == 0:
            return None, None
        model = self.Model_Constructor(
            fit_intercept=True,
            normalize=True,
            copy_X=True)
        model.fit(train_x, train_y)
        # if the model cannot be fit, LinearRegression returns the mean
        # of the train_y values
        predictions = model.predict(test_x)
        return predictions, model

    def run(self, train, test, control):
        '''fit on training data and test

        ARGS
        train  : dataframe
        test   : dataframe
        control: Bunch

        RETURN
        dict with key = (x_mode, y_mode) values = (actuals, estimates, fitted)
        '''
        # implement variants
        verbose = True
        debug = True
        if debug:
            print 'OLS.run debug'
        all_variants = {}
        for x_mode in ('linear',) if debug else ('log', 'linear'):
            for y_mode in ('log',) if debug else ('log', 'linear'):
                train_x, x_names = x(x_mode, train, control)
                test_x, _ = x(x_mode, test, control)
                train_y = y(y_mode, train, control)
                model = self.Model_Constructor(fit_intercept=True,
                                               normalize=True,
                                               copy_X=True)
                fitted_model = model.fit(train_x, train_y)
                # if the model cannot be fitted, LinearRegressor returns
                # the mean of the train_y values
                estimates = fitted_model.predict(test_x)
                if debug:
                    print 'train_x.shape', train_x.shape
                    if fitted_model is not None:
                        print 'coef_', fitted_model.coef_
                        print 'estimates', estimates
                key = ('y_mode', y_mode, 'x_mode', x_mode)
                value = {
                    'model': fitted_model,  # contains coefficient and intercept
                    'x_names': x_names,
                    'estimates': demode(estimates, y_mode),
                    'actuals': y('linear', test, control)
                }
                # check results
                if verbose:
                    print 'x_mode, y_mode: ', x_mode, y_mode
                    print 'actuals: ', value['actuals']
                    print 'estimates: ', value['estimates']
                all_variants[key] = value
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


def report_global(key, result):
    print 'report global', key, result


def report_zip_code(key, result):
    print 'report zip code', key, result


def report(sale_date, training_days, model_name, scope, run_result, control):
    'print report for given selectors'

    print_folds = True
    n_folds = control.n_folds

    def median_abs_error(actuals, estimates):
        abs_error = np.abs(actuals - estimates)
        median_abs_error = np.median(abs_error)
        return median_abs_error

    format_global_fold = '%10s %2d %3s %6s %3s %3s f%d %6.0f %3.2f'
    format_zip_fold = '%10s %2d %3s %6d %3s %3s f%d %6.0f %3.2f'
    format_global = '%10s %2d %3s %6s %3s %3s median %6.0f %3.2f'
    format_zip = '%10s %2d %3s %6d %3s %3s median %6.0f %3.2f'

    def print_scope_global():
        for x_mode in ('log', 'linear'):
            for y_mode in ('log', 'linear'):
                errors = np.zeros(n_folds, dtype=np.float64)
                rel_errors = np.zeros(n_folds, dtype=np.float64)
                for fold_number in xrange(n_folds):
                    key = (
                        sale_date,
                        training_days,
                        model_name,
                        scope,
                        fold_number)
                    model_run = run_result[key]
                    model_run_key = (
                        'x_mode',
                        x_mode,
                        'y_mode',
                        y_mode)
                    model_run_value = model_run[model_run_key]
                    actuals = model_run_value['actuals']
                    estimates = model_run_value['estimates']
                    error = median_abs_error(actuals, estimates)
                    rel_error = error / np.median(actuals)
                    if print_folds:
                        line = format_global_fold % (
                            sale_date,
                            training_days,
                            model_name,
                            scope,
                            y_mode[:3],
                            x_mode[:3],
                            fold_number,
                            error,
                            rel_error,
                        )
                        print line  # result for one fold
                    # accumulate across folds
                    errors[fold_number] = error
                    rel_errors[fold_number] = rel_error
                line = format_global % (
                    sale_date,
                    training_days,
                    model_name,
                    scope,
                    y_mode[:3],
                    x_mode[:3],
                    np.median(errors),
                    np.median(rel_errors))
                print line

#    def print_scope_local2():
#        pdb.set_trace()
#        # find the zip code-based results
#        for fold_number in xrange(n_folds):
#            fold_run_result = run_result[(sale_date,
#                                          training_days,
#                                          model_name,
#                                          scope,
#                                          fold_number)]
#            for zip_code, zip_code_result in fold_run_result.iteritems():
#                print zip_code
#                for x_mode in ('log', 'linear'):
#                    for y_mode in ('log', 'linear'):
#                        zip_code_result_key = (
#                            'x_mode', x_mode,
#                            'y_mode', y_mode)
#                        model_run = zip_code_result[zip_code_result_key]

    def get_all_zip_codes():
        'return zip-codes in every fold'
        # zip-codes may differe across folds
        all_zip_codes_by_fold_number = collections.defaultdict(set)
        for fold_number in xrange(n_folds):
            a_run_result = run_result[(sale_date,
                                       training_days,
                                       model_name,
                                       scope,
                                       fold_number)]
            for zip_code in a_run_result.keys():
                all_zip_codes_by_fold_number[fold_number].add(zip_code)
        zip_codes_in_all_folds = all_zip_codes_by_fold_number[0]
        for zip_codes in all_zip_codes_by_fold_number.values():
            zip_codes_in_all_folds.intersection(zip_codes)
        return zip_codes_in_all_folds

    def zip_code_run_result(zip_code):
        'retun new run_results containing just items for the zip code'
        print zip_code
        result = {}
        for k, v in run_result.iteritems():
            date_time, training_days, model_name, scope, fold_number = k
            if scope != 'zip':
                continue
            for run_result_key, run_result_value in v.iteritems():
                if run_result_key == zip_code:
                    result.append(run_result_value)
                pdb.set_trace()
                pass
        pass

    def print_scope_zip():
        # determine all the zip code
        all_zips = get_all_zip_codes()
        sorted_zips = sorted(all_zips)
        for zip_code in sorted_zips:
            for x_mode in ('log', 'linear'):
                for y_mode in ('log', 'linear'):
                    errors = np.zeros(n_folds, dtype=np.float64)
                    rel_errors = np.zeros(n_folds, dtype=np.float64)
                    for fold_number in xrange(n_folds):
                        pdb.set_trace()
                        key = (sale_date,
                               training_days,
                               model_name,
                               'zip',
                               fold_number)
                        print key
                        model_run = run_result[key][zip_code][
                            ('x_mode', x_mode, 'y_mode', y_mode)]
                        actuals = model_run['actuals']
                        estimates = model_run['estimates']
                        error = median_abs_error(actuals, estimates)
                        rel_error = error / np.median(actuals)
                        if print_folds:
                            line = format_zip_fold % (sale_date,
                                                      training_days,
                                                      model_name,
                                                      zip_code,
                                                      y_mode[:3],
                                                      x_mode[:3],
                                                      fold_number,
                                                      error,
                                                      rel_error)
                            print line
                        errors[fold_number] = error
                        rel_errors[fold_number] = rel_error
                    pdb.set_trace()
                    line = format_zip % (sale_date,
                                         training_days,
                                         model_name,
                                         zip_code,
                                         y_mode[:3],
                                         x_mode[:3],
                                         np.median(errors),
                                         np.median(rel_errors))
                    print line

    debug = True
    if debug:
        print 'bypassing printing of report'
        return
    if scope == 'global':
        print_scope_global()
    elif scope == 'zip':
        print 'bypassing print_scope_zip'
        return
        print_scope_zip()
    else:
        print scope
        raise RuntimeError('bad scope: ' + str(scope))


def make_train_model(df, sale_date, training_days):
    'return df of transactions no more than training_days before the sale_date'
    just_before_sale_date = within(sale_date, training_days, df)
    train_model = add_age(train[just_before_sale_date], sale_date)
    return train_model


def make_test_model(df, sale_date):
    'return df of transactions on the sale_date'
    selected_indices = on_sale_date(sale_date, test)
    test_model = add_age(test[selected_indices], sale_date)
    return test_model


# MAIN PROGRAM
# convert warnings into errors
warnings.filterwarnings('error')

# read command line and set control variables

control = make_control(sys.argv)

sys.stdout = Logger(logfile_path=control.path_log)
print control

# read training data

print "reading training data"
f = open(control.path_in, 'rb')
df_loaded = pickle.load(f)
print df_loaded.shape
df_loaded_copy = df_loaded.copy(deep=True)
f.close()

KFold = cross_validation.KFold
run_result = {}

verbose = True
debug = True
last_train_indices = np.array([])

# make sure each sale_date sees the same fold data
kf = KFold(n=(len(df_loaded)),
           n_folds=control.n_folds,
           shuffle=True,
           random_state=control.random_seed)
all_results = {}
fold_number = -1
pdb.set_trace()
for train_indices, test_indices in kf:
    fold_number += 1

    # don't create views (just to be careful)
    train = df_loaded.iloc[train_indices].copy(deep=True)
    test = df_loaded.iloc[test_indices].copy(deep=True)
    assert(df_loaded.equals(df_loaded_copy))

    for sale_date in control.sale_dates:
        for training_days in (7, 14) if debug else control.training_days:
            train_model = make_train_model(train, sale_date, training_days)
            test_model = make_test_model(test, sale_date)
            for model_name, model in control.models.iteritems():

                def make_key(last_item):
                    return (fold_number, sale_date, training_days,
                            model_name, last_item)

                # determine results for all areas (i.e., global)
                global_result = model.run(train=train_model,
                                          test=test_model,
                                          control=control)
                key = make_key('global')
                all_results[key] = global_result
                report_global(key, global_result)

                # determine results for each zip code in test data
                for zip_code in unique_zip_codes(test_model):
                    print 'zip_code', zip_code
                    train_model_zip = zip_codes(train_model, zip_code)
                    test_model_zip = zip_codes(test_model, zip_code)
                    # Note: there can be no training data
                    if len(train_model_zip) > 0:
                        # there is  training date for the zip code
                        zip_code_result = model.run(train=train_model_zip,
                                                    test=test_model_zip,
                                                    control=control)
                        key = make_key(('zip', zip_code))
                        all_results[key] = zip_code_result
                        report_zip_code(key, zip_code_result)
print 'finished loop over folds'
pdb.set_trace()

result = {'control': control,
          'run_result': run_result}
# TODO: write result into file system
