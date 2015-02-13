# create file WORKING/CV-CELL/<command line argument>
#
# COMMAND LINE ARGUMENT is one positional in this format:
# MODEL-RESPONSE-PREDICTORS-YEARS-DAYS
# where
# MODEL is one of {ols}
# RESPONSE is one of {price, logprice}
# PREDICTORS is an argument to function features
# TESTYEARS is one of {2008, 2003on}
# TRAININGDAYS is one of {30, 60, ..., 360}

# import built-ins and libraries
import sys
import pdb
import cPickle as pickle
from sklearn import cross_validation
from sklearn import linear_model
import sklearn
import numpy as np
import datetime
import pandas as pd

# import my stuff
from directory import directory
from features import features
from Logger import Logger
from CvResult import CvResult
from FoldResult import FoldResult


def print_calling_sequence():
    print 'argument: RESPONSE-PREDICTORS-MODEL-YEAR-NDAYS'


class Control(object):
    def __init__(self, arguments):
        me = 'cv-cell'

        working = directory('working')
        log = directory('log')
        cvcell = working + 'cv-cell/'

        # confirm that exactly one argument
        if len(arguments) != 2:
            print_calling_sequence()
            raise RuntimeError('need exactly one positional argument')

        # parse the positional argument
        arg1 = arguments[1]
        splits = arg1.split('-')
        self.model = splits[0]
        self.response = splits[1]
        self.predictors = splits[2]
        self.test_years = splits[3]
        self.training_days = splits[4]

        # make sure that PREDICTORS is known
        try:
            features(self.predictors)
        except RuntimeError:
            print_calling_sequence()
            raise RuntimeError('unknown predictors:' + self.predictors)

        self.path_out = cvcell + arg1 + '.pickle'
        self.path_out_log = log + me + arg1 + '.log'
        self.path_in_train = working + 'transactions-subset2-train.pickle'

        self.command_line = arg1
        self.n_folds = 10

        # make random numbers reproducable
        self._random_seed = 123  # don't use this, define just for documentation
        # create np.random.RandomState instance
        self.random_state = sklearn.utils.check_random_state(self._random_seed)

        self.testing = False
        self.debugging = False
        # eliminate null is input column sale.python_date
        self.debugging_sale_python_date = False


def relevant_test(df, test_years):
    ''' Return DataFrame containing just the relevant test transactions.'''
    if test_years == '2008':
        in_testing = df['sale.year'] == 2008
    else:
        raise NotImplemented('test_years: ' + test_years)
    return df[in_testing]


def relevant_train(df, the_sale_datetime, training_days):
    '''Return rows within training_days of the sale date.'''
    pdb.set_trace()
    days_before = the_sale_datetime - df['sale.datetime']
    in_training = days_before <= training_days
    return df[in_training]


def add_ageOLD(the_sale_datetime, df):
    '''Mutate df to have age and age^2 features.

    Do this if the df has features year built or effective year built.
    '''

    def add(new_column_name_base, year_column_name):
        'Add age and age^2, in years.'
        pdb.set_trace()
        year = df[year_column_name]
        year_datetime = np.datetime64(year)
        age = the_sale_datetime - year_datetime
        age2 = age * age
        df[new_column_name_base] = age
        df[new_column_name_base + '2'] = age2

    pdb.set_trace()
    column_names = df.columns
    if 'year.built' in column_names:
        add('age', 'year.built')
    if 'effective.year.built' in column_names:
        add('effective.age', 'effective.year.built')


def check_sale_datetime(dt):
    '''Check that the time components are always zero.'''
    # NOTE: This code doesn't work, as there are not such fields
    # MAYBE FIX: use pd.DateTimeIndex when creating
    pdb.set_trace()
    ok = dt.hour == 0 and \
        dt.minute == 0 and \
        dt.second == 0 and \
        dt.microsecond == 0
    if not ok:
        print 'datetime with non-zero time element', dt


def actuals_estimates(sale_date, fold_test, fold_train, control):
    '''Return actuals and estimates for all test transactions with sale date.

    Return two np arrays.
    '''
    # check_sale_datetime(sale_datetime)

    # create test data
    test_indices = np.where(fold_test['sale.python_date'] == sale_date)
    testing = fold_test.iloc[test_indices]
    actuals = testing['SALE.AMOUNT']

    # create training data
    train_date = fold_train['sale.python_date']
    training_days = datetime.timedelta(int(control.training_days))
    before_sale = train_date < sale_date
    within_training_days = (train_date + training_days) >= sale_date
    in_training = np.logical_and(before_sale, within_training_days)
    if in_training.sum() > 3000:  # while debugging
        print in_training.sum()
        print 'big training set'
        pdb.set_trace()
    training = fold_train[in_training]

    train_x, train_y, test_x = xy(sale_date, training, testing, control)

    if control.model == 'ols':
        m = linear_model.LinearRegression(fit_intercept=True,
                                          normalize=False,
                                          copy_X=True)
        m.fit(train_x, train_y)
        estimates = m.predict(test_x)
    else:
        raise NotImplemented('model: ' + control.model)

    return actuals, estimates


def response(df, response_name):
    '''Return numpy 1D array with selected column.'''
    values = np.array(df['SALE.AMOUNT'])
    if response_name == 'price':
        return values
    elif response_name == 'logprice':
        return np.log(values)
    else:
        raise NotImplemented('response: ', response_name)


def predictors(df, predictor_names):
    '''Return numpy 2D array with selected columns.'''
    # build the result matrix m in transposed form
    m = np.empty([len(predictor_names), df.shape[0]])
    column_index = 0
    for column_name in predictor_names:
        values = df[column_name]
        m[column_index] = values
        column_index += 1

    return m.transpose()


def maybe_add_age(df_list, from_to_list, predictor_names, test_date):
    'If DataFrame contains year.built or effective.year.built, add age, age^2.'

    def column_names(from_to):
        name_from, name_to = from_to
        return [name_from, name_to, name_to + '2']

    def helper(df, from_to):
        year_name, age_name, age2_name = column_names(from_to)
        if year_name in df.columns:
            year = df[year_name]
            test_year = pd.to_datetime([test_date]).year
            age = test_year - year  # age in whole number of years
            df[age_name] = age
            df[age2_name] = age * age

    # add age column to DataFrames
    for df in df_list:
        for from_to in from_to_list:
            helper(df, from_to)

    # adjust column names in predictors
    for from_to in from_to_list:
        year_name, age_name, age2_name = column_names(from_to)
        predictor_names.remove(year_name)
        predictor_names.append(age_name)
        predictor_names.append(age2_name)


def xy(test_date, training, testing, control):
    '''Return train_x. train_y, test_x.'''
    predictor_names = features(control.predictors).keys()

    maybe_add_age([training, testing],
                  [['YEAR.BUILT', 'age'],
                   ['EFFECTIVE.YEAR.BUILT', 'effective.age']],
                  predictor_names,
                  test_date)

    train_x = predictors(training, predictor_names)
    test_x = predictors(testing, predictor_names)
    train_y = response(training, control.response)

    return train_x, train_y, test_x


def fit_local_models(df, control):
    '''Return CvResult instance for the cell specified by control.command_line.
    '''

    # create iterator across fold indices
    kf = cross_validation.KFold(df.shape[0],
                                n_folds=control.n_folds,
                                shuffle=True,
                                random_state=control.random_state)

    # for each fold
    cvresult = CvResult()
    fold_number = 0
    for train_indices, test_indices in kf:
        fold_number += 1

        # split df into train and test
        fold_train = df.iloc[train_indices]
        fold_test = df.iloc[test_indices]

        fold_test_relevant = relevant_test(fold_test, control.test_years)

        # iterate over the unique sale dates in the test fold
        # sort the dates, so that the print out is easier to follow visually
        dates = fold_test_relevant['sale.python_date']
        unique_sale_dates = dates.unique()
        unique_sale_dates_sorted = np.sort(unique_sale_dates)

        # create the fold results by analyzing each date
        foldresult = FoldResult()
        for this_sale_date in unique_sale_dates_sorted.flat:
            actuals, estimates = actuals_estimates(this_sale_date,
                                                   fold_test_relevant,
                                                   fold_train,
                                                   control)
            foldresult.extend(actuals, estimates)
            print \
                control.command_line, \
                fold_number, \
                this_sale_date, \
                len(estimates)

        cvresult.save_FoldResult(foldresult)

    return cvresult


def main():

    control = Control(sys.argv)
    sys.stdout = Logger(logfile_path=control.path_out_log)

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    # read training data
    f = open(control.path_in_train, 'rb')
    df = pickle.load(f)
    f.close()

    # check that no sale.python_date value is not null
    if control.debugging_sale_python_date:
        dates = df['sale.python_date']
        if dates.isnull().any():
            print dates
            raise ValueError('a sale.python_date is null')

    cv_result = fit_local_models(df=df,
                                 control=control)

    # write cross validation result
    f = open(control.path_out, 'wb')
    pickle.dump((cv_result, control), f)
    f.close()

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    print 'done'

if __name__ == '__main__':
    main()
