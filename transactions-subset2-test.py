# create files
# WORKING/transactions-subset2-train.pickle
# WORKING/transactions-subset2-test.pickle
#
# The test data consists of a 10% random sample of all the data.
#
# Unlike the R version, the data are not stratified by sale month.

# import built-ins and libraries
import numpy as np
import pandas as pd
import pdb
import sys
from sklearn import cross_validation

# import my stuff
from directory import directory
from Logger import Logger


class Control(object):
    def __init__(self):
        me = 'transactions-subset2-test'
        working = directory('working')
        log = directory('log')

        self.path_out_test = working + me + '-test.pickle'
        self.path_out_train = working + me + '-train.pickle'
        self.path_out_log = log + me + '.log'
        self.path_in_data = working + 'transactions-subset2.pickle'

        self.test_sample = .10
        self.random_seed = 123

        self.testing = False


def randomly_split(df, fraction_to_testing, random_state):
    '''Randomly shuffly observations and split into train and test.'''

    ss = cross_validation.ShuffleSplit(n=df.shape[0],
                                       n_iter=1,
                                       test_size=fraction_to_testing,
                                       random_state=random_state)

    # extract the train and test indices
    # there should be exactly one set of such
    num_iterations = 0
    for train_index, test_index in ss:
        num_iterations = num_iterations + 1
        train_indices = train_index
        test_indices = test_index
    assert num_iterations == 1

    test = df.iloc[test_indices]
    train = df.iloc[train_indices]

    return test, train


def analyze(test, train):
    print 'test.shape', test.shape
    print 'train.shape', train.shape

    # check range of sale.year
    min_year = min(np.amin(test['sale.year']),
                   np.amin(train['sale.year']))
    max_year = max(np.amax(test['sale.year']),
                   np.amax(train['sale.year']))
    assert min_year == 2003
    print max_year
    assert max_year == 2009

    print 'year, month, #test, #train'
    for year in (2003, 2004, 2005, 2006, 2007, 2008, 2009):
        last_month_index = 12 if year != 2009 else 3
        for month_index in range(last_month_index):
            month = month_index + 1
            is_month_test = np.logical_and(test['sale.year'] == year,
                                           test['sale.month'] == month)
            in_month_test = test[is_month_test]
            is_month_train = np.logical_and(train['sale.year'] == year,
                                            train['sale.month'] == month)
            in_month_train = train[is_month_train]
            print year, month, in_month_test.shape[0], in_month_train.shape[0]


def main():

    control = Control()
    sys.stdout = Logger(logfile_path=control.path_out_log)

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    df = pd.read_pickle(control.path_in_data)

    if control.testing and False:
        pdb.set_trace()
        df = df[0:30]

    # print columns in df
    print 'df.shape', df.shape
    for column_name in df.columns:
        print 'df column name', column_name

    test, train = randomly_split(df=df,
                                 fraction_to_testing=control.test_sample,
                                 random_state=control.random_seed)
    analyze(test, train)

    test.to_pickle(control.path_out_test)
    train.to_pickle(control.path_out_train)

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v
    print 'done'

if __name__ == '__main__':
    main()
