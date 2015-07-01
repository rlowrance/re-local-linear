# randomly split a transactions file into test and train portions
# the transactions file is pickled and contains a Pandas DataFrame object
# the split files are also pickled and each contains a Pandas DataFrame object
# invocation options:
#  --test=FRACTION       : fraction to test, default is 0.10
#  --in PATHIN           : path to input transaction pickle file
#  --outtest PATHTEST    : path to test file (.pickle)
#  --outtrain PATHTRAIN  : path to training file (.pickle)
#
# The test data consists of a 10% random sample of all the data.
#
# Unlike the R version, the data are not stratified by sale month.

# import built-ins and libraries
import numpy as np
import datetime
import pandas as pd
import pdb
import sys
from sklearn import cross_validation
import warnings

# import my stuff
from Bunch import Bunch
from directory import directory
from Logger import Logger
import parse_command_line


def make_control(argv):
    # return a Bunch
    script_name = argv[0]

    b = Bunch(debugging=False,
              testing=False,
              now=datetime.datetime.now(),
              base_name=script_name.split('.')[0],
              me=script_name,
              random_seed=123,
              arg_fraction=parse_command_line.default(argv, '--fraction', 0.10),
              arg_in=parse_command_line.get_arg(argv, '--in'),
              arg_outtest=parse_command_line.get_arg(argv, '--outtest'),
              arg_outtrain=parse_command_line.get_arg(argv, '--outtrain'))
    return b


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
    warnings.filterwarnings('error')
    control = make_control(sys.argv)
    path = \
        directory('log') + \
        control.base_name + '.' + control.now.isoformat('T') + '.log'
    sys.stdout = Logger(logfile_path=path)  # print x now logs and prints x
    print control

    df = pd.read_pickle(control.arg_in)

    # make sure that sale.python_date is never NaN (null)
    if False:
        dates = df['sale.python_date']
        if dates.isnull().any():
            raise ValueError('at least one sale.python_date is null')

    if control.testing and False:
        pdb.set_trace()
        df = df[0:30]

    # print columns in df
    print 'df.shape', df.shape
    for column_name in df.columns:
        print 'df column name', column_name

    test, train = randomly_split(df=df,
                                 fraction_to_testing=control.arg_fraction,
                                 random_state=control.random_seed)
    analyze(test, train)

    test.to_pickle(control.arg_outtest)
    train.to_pickle(control.arg_outtrain)

    # log the control variables
    print control
    print 'done'

if __name__ == '__main__':
    main()
