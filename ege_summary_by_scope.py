'''create WORKING/ege_summary_by_scope-YYYY-MM-DD.pickle

INPUT FILES: WORKING/ege_date-YYYY-MM-DD.dataframe.pickle
OUTPUT FILES: WORKING/chart-05.txt
 contains analysis of each quarter
'''

import cPickle as pickle
import datetime
import numpy as np
import operator
import pandas as pd
from pprint import pprint  # for debugging
import random
import pdb
import sys

# import my stuff
from Bunch import Bunch
from DataframeAppender import DataframeAppender
from directory import directory
from Logger import Logger
import parse_command_line
from ege_date import Rf, Ols, ReportRf, ReportOls


def print_help():
    print 'python ege_summary_by_scope.py YYYY-MM-DD'


def make_control(argv):
    # return a Bunch

    random.seed(123456)
    base_name = argv[0].split('.')[0]
    now = datetime.datetime.now()
    testing = False

    sale_date = argv[1]

    b = Bunch(debugging=False,
              path_log=directory('log') + base_name + '.' + now.isoformat('T') + '.log',
              path_in=directory('working') + ('ege_to_dataframe-%s.pickle' % sale_date),
              path_out=directory('working') + ('ege_summary_by_scope-%s.pickle' % sale_date),
              testing=testing)

    return b


def median(lst):
    return np.median(np.array(lst, dtype=np.float64))


def summarize(df, control):
    'determine model performance across folds in each scope, model, training days'

    reduced_df = DataframeAppender([('scope', object),
                                    ('model_id', object),
                                    ('training_days', np.float64),
                                    ('num_folds', np.float64),
                                    ('median_abs_error', np.float64),
                                    ('median_rel_error', np.float64)])

    def unique(column_name):
        'return sorted unique elements in column'
        return sorted(pd.Series.unique(df[column_name]))

    def errors(scope, model_id, training_days):
        'return median abs error, median rel error, num folds with estimates'
        verbose = False
        all_abs_errors = []
        all_rel_errors = []
        for fold_number in unique('fold_number'):
            df1 = df[(df.scope == scope) &
                     (df.model_id == model_id) &
                     (df.training_days == training_days) &
                     (df.fold_number == fold_number)]
            if len(df1) == 0:
                if verbose:
                    print 'no data in fold', scope, model_id, training_days, fold_number
                continue
            assert len(df1) == 1
            all_abs_errors.append(df1.abs_error.iloc[0])
            all_rel_errors.append(df1.rel_error.iloc[0])
        if len(all_rel_errors) == 0:
            print 'no data in all folds', scope, model_id, training_days
            return None, None, 0
        median_all_abs_errors = median(all_abs_errors)
        median_all_rel_errors = median(all_rel_errors)
        if median_all_abs_errors > 1e7:
            print 'large abs error', median_all_abs_errors, all_abs_errors
        if median_all_rel_errors > 1:
            print 'large rel error', median_all_rel_errors, all_rel_errors
        return median_all_abs_errors, median_all_rel_errors, len(all_rel_errors)

    verbose = True
    for scope in unique('scope'):
        for model_id in unique('model_id'):
            for training_days in unique('training_days'):
                median_abs_error, median_rel_error, num_folds = errors(scope, model_id, training_days)
                if median_abs_error is not None and median_rel_error is not None:
                    reduced_df.append(
                        [scope, model_id, training_days, num_folds, median_abs_error, median_rel_error])
                    if verbose:
                        print '%6s %11s %3d %2d %6.0f %4.2f' % (
                            scope, model_id, training_days, num_folds, median_abs_error, median_rel_error
                        )

    return reduced_df.result()


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_log)
    print control

    f = open(control.path_in, 'rb')
    df = pickle.load(f)
    f.close()

    reduced_df = summarize(df, control)

    f = open(control.path_out, 'wb')
    pickle.dump(reduced_df, f)
    f.close()

    print control
    if control.testing:
        print 'DISCARD OUTPUT: TESTING'
    print 'done'

    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        Ols()
        ReportOls()
        Rf()
        ReportRf()
        parse_command_line()
        pd.DataFrame()
        np.array()
        operator.add()

    main(sys.argv)
