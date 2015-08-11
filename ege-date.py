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

    df2 = df.copy(deep=True)
    if mode == 'log':
        # some features are transformed to log, some to log1p, some not at all
        for predictor_name, transformation in control.predictors.iteritems():
            if transformation == 'log':
                df2[predictor_name] = \
                    pd.Series(np.log(df[predictor_name]),
                              index=df.index)
            elif transformation == 'log1p':
                df2[predictor_name] = \
                    pd.Series(np.log1p(df[predictor_name]),
                              index=df.index)
            elif transformation is None:
                pass
            else:
                raise RuntimeError('bad transformation: ' + transformation)
    selected_columns = control.predictors.keys()
    # force dtype, otherwise will be dtype=object
    array = np.array(df2[selected_columns].values, np.float64)
    return array, selected_columns


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

    def fit_and_predict(self, train_x, train_y, test_x):
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

    def run_global(self, train, test, control):
        '''fit and test one model for all the samples

        RETURN
        dict with key = (x_mode, y_mode)
        '''
        # implement variants
        verbose = False
        all_variants = {}
        for x_mode in ('log', 'linear'):
            for y_mode in ('log', 'linear'):
                train_x, x_names = x(x_mode, train, control)
                test_x, _ = x(x_mode, test, control)
                train_y = y(y_mode, train, control)
                estimates, fitted_model = self.fit_and_predict(
                    train_x=train_x,
                    train_y=train_y,
                    test_x=test_x)
                key = ('x_mode', x_mode, 'y_mode', y_mode)
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

    def run_zip(self, train, test, control):
        'fit and test a model for each zip5 in the test samples'
        verbose = False
        zip_results = {}
        zip_uniques = test.zip5.unique()
        for i in xrange(len(zip_uniques)):
            # zip5 is a scalar numpy value
            zip5 = zip_uniques[i]
            if verbose:
                print zip5, type(zip5)
            global_result = self.run_global(
                train.loc[train.zip5 == zip5].copy(deep=True),
                test.loc[test.zip5 == zip5].copy(deep=True),
                control)
            zip_results[zip5] = global_result
            if verbose:
                print 'zip5: ', zip5
                for key, value in global_result.iteritems():
                    print key
                    print 'estimates: ', value['estimates']
                    print 'actuals: ', value['actuals']
        return zip_results

    def run(self, train, test, scope, control):
        'return dict of variants of fited model and predict'
        if scope == 'global':
            return self.run_global(
                train.copy(deep=True),
                test.copy(deep=True),
                control)
        elif scope == 'zip':
            return self.run_zip(
                train.copy(deep=True),
                test.copy(deep=True),
                control)
        else:
            print 'bad scope: ' + scope
            raise RuntimeError()


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

    b = Bunch(
        path_in=directory('working') + 'transactions-subset2.pickle',
        path_log=directory('log') + log_file_name,
        arg_date=sale_date,
        random_seed=random_seed,
        sale_dates=[sale_date],
        models={'ols': Ols()},
        scopes=['global', 'zip'],
        training_days=range(7, 366, 7),
        n_folds=10,
        predictors=predictors,
        price_column='SALE.AMOUNT',
        debug=False)
    return b


def within_training_days(sale_date, training_days, df):
    'Return df containing only samples within training_days days of sale_date'
    first_ok_sale_date = sale_date - datetime.timedelta(training_days)
    date_column = 'sale.python_date'
    after = df[date_column] > first_ok_sale_date
    before = df[date_column] <= sale_date
    ok_indices = np.logical_and(after, before)
    ok_df = df.loc[ok_indices]  # mask selection
    return ok_df


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


def report(sale_date, training_days, model_name, scope, run_result, control):
    'print report for given selectors'

    print_folds = True
    n_folds = control.n_folds

    def median_abs_error(actuals, estimates):
        abs_error = np.abs(actuals - estimates)
        median_abs_error = np.median(abs_error)
        return median_abs_error

    format_global_fold = '%10s %2d %3s %6s %3s %3s f%02d %6.0f %3.2f'
    format_zip_fold = '%10s %2d %3s %6d %3s %3s f%02d %6.0f %3.2f'
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

    def print_scope_local2():
        pdb.set_trace()
        # find the zip code-based results
        for fold_number in xrange(n_folds):
            fold_run_result = run_result[(sale_date,
                                          training_days,
                                          model_name,
                                          scope,
                                          fold_number)]
            for zip_code, zip_code_result in fold_run_result.iteritems():
                print zip_code
                for x_mode in ('log', 'linear'):
                    for y_mode in ('log', 'linear'):
                        zip_code_result_key = (
                            'x_mode', x_mode,
                            'y_mode', y_mode)
                        model_run = zip_code_result[zip_code_result_key]

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

    def print_scope(scope, format_fold, format):
        for x_mode in ('log', 'linear'):
            for y_mode in ('log', 'linear'):
                errors = np.zeros(n_folds, dtype=np.float64)
                rel_errors = np.zeros(n_folds, dtype=np.float64)
                for fold_number in xrange(n_folds):
                    pdb.set_trace()
                    key = (sale_date,
                           training_days,
                           model_name,
                           scope,
                           fold_number)
                    model_run = run_result[key]
                    actuals = model_run['actuals']
                    estimates = model_run['estimates']
                    error = median_abs_error(actuals, estimates)
                    rel_error = error / np.median(actuals)
                    if print_folds:
                        line = format_fold % (sale_date,
                                              training_days,
                                              model_name,
                                              scope,
                                              y_mode[:3],
                                              x_mode[:3],
                                              fold_number,
                                              error,
                                              rel_error)
                        print line
                    errors[fold_number] = error
                    rel_errors[fold_number] = rel_error
                line = format % (sale_date,
                                 training_days,
                                 model_name,
                                 scope,
                                 y_mode[:3],
                                 x_mode[:3],
                                 np.median(errors),
                                 np.median(rel_errors))
                print line

    def print_scope_zip():
        # determine all the zip code
        return
        all_zips = get_all_zip_codes()
        sorted_zips = sorted(all_zips)
        for zip_code in sorted_zips:
            print_scope(zip_code, format_zip_fold, format_zip)

    if scope == 'global':
        print_scope_global()
    elif scope == 'zip':
        print_scope_zip()
    else:
        print scope
        raise RuntimeError('bad scope: ' + str(scope))


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
loaded_df = pickle.load(f)
f.close()
print 'loaded_df shape', loaded_df.shape

KFold = cross_validation.KFold
run_result = {}

verbose = False
last_train_indices = np.array([])
for sale_date in control.sale_dates:
    df = loaded_df.copy(deep=True)  # we mutate df, so start with a fresh copy
    for training_days in control.training_days:
        df = within_training_days(sale_date, training_days, df.copy(deep=True))
        if verbose:
            print sale_date, training_days
        df_aged = add_age(df, sale_date)
        for model_name, model in control.models.iteritems():
            for scope in control.scopes:
                kf = KFold(n=len(df_aged),
                           n_folds=control.n_folds,
                           shuffle=True,
                           random_state=control.random_seed)
                fold_number = 0
                for train_indices, test_indices in kf:
                    if verbose:
                        print \
                            sale_date, training_days, model_name, scope, \
                            fold_number
                    # check that folds are actually formed
                    if np.array_equal(last_train_indices, train_indices):
                        print 'BAD train_indices'
                        pdb.set_trace()
                    last_train_indices = train_indices.copy()
                    train = df_aged.iloc[train_indices].copy()
                    test = df_aged.iloc[test_indices].copy()
                    assert len(train) + len(test) == len(df_aged)
                    model_run = model.run(
                        train=train,
                        test=test,
                        scope=scope,
                        control=control)
                    key = (
                        sale_date,
                        training_days,
                        model_name,
                        scope,
                        fold_number)
                    run_result[key] = model_run
                    fold_number += 1
                report(
                    sale_date,
                    training_days,
                    model_name,
                    scope,
                    run_result,
                    control)
                print
result = {'control': control,
          'run_result': run_result}
# TODO: write result into file system
