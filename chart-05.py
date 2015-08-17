'''create WORKING/chart-05-X.txt files

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
from directory import directory
from Logger import Logger
import parse_command_line
from Report import Report
from ege_date import Rf, Ols, ReportRf, ReportOls


def print_help():
    print 'python chart-05.py --in PATH [--cache]'
    print 'where'
    print ' PATH is the path to the input file'
    print ' --cache reads from the cache instead of PATH'
    print '     the cache contains a reduction of the input'


def make_control(argv):
    # return a Bunch

    random.seed(123456)
    base_name = argv[0].split('.')[0]
    now = datetime.datetime.now()
    testing = True

    in_file_dates = ('2009-02-16',) if testing else None  # FIXME: use real list

    b = Bunch(debugging=False,
              path_log=directory('log') + base_name + '.' + now.isoformat('T') + '.log',
              path_in_base=directory('working') + 'ege_to_dataframe-%s.pickle',
              path_out=directory('working') + base_name + '.txt',
              in_file_dates=in_file_dates,
              testing=testing)

    return b


def median(lst):
    return np.median(np.array(lst, dtype=np.float64))


def analyze(sale_date, df, report, control):
    'append to report'

    class Reduced(object):
        def __init__(self, names, dtypes):
            assert(len(names) == len(dtypes))
            self.names = names
            self.dtypes = dtypes
            arg = []
            for _ in names:
                arg.append(None)
            self.df = self._make(arg)

        def append(self, lst):
            assert(len(lst) == len(self.names))
            new_df = self._make(lst)
            self.df = self.df.append(new_df, ignore_index=True)

        def result(self):
            return self.df

        def _make(self, lst):
            def maybe(x):
                return [] if x is None else [x]

            def s(x, dtype):
                return pd.Series(maybe(x), dtype=dtype)

            assert(len(lst) == len(self.names))
            d = {}
            for i in xrange(len(self.names)):
                d[self.names[i]] = s(lst[i], self.dtypes[i])
            return pd.DataFrame(d)

    report.append('sale_date: %s' % sale_date)

    # determine the best models
    reduced_df = Reduced(names=('scope', 'model_id', 'training_days', 'median_rel_error', 'median_abs_error'),
                         dtypes=(object, object, np.float64, np.float64, np.float64))
    format_fold_none = '%6s %11s %3d fold %1d none'
    format_fold_some = '%6s %11s %3d fold %1d %6.0f %4.2f'
    format_acrs_none = '%6s %11s %3d median none'
    format_acrs_some = '%6s %11s %3d median %6.0f %4.2f'
    # for scope in pd.Series.unique(df.scope):
    verbose = False
    for scope in ('global',):
        for model_id in pd.Series.unique(df.model_id):
            for training_days in sorted(pd.Series.unique(df.training_days)):
                abs_errors = []
                rel_errors = []
                for fold_number in sorted(pd.Series.unique(df.fold_number)):
                    df1 = df[(df.scope == scope) &
                             (df.model_id == model_id) &
                             (df.training_days == training_days) &
                             (df.fold_number == fold_number)]
                    training_days = int(training_days)
                    fold_number = int(fold_number)
                    if len(df1) == 0:
                        if verbose:
                            print format_fold_none % (scope, model_id, training_days, fold_number)
                        continue
                    assert(len(df1) == 1)
                    abs_error = df1.abs_error.iloc[0]
                    rel_error = df1.rel_error.iloc[0]
                    abs_errors.append(abs_error)
                    rel_errors.append(rel_error)
                    if verbose:
                        print format_fold_some % (scope, model_id, training_days, fold_number,
                                                  abs_error, rel_error)
                if len(abs_errors) == 0:
                    if verbose:
                        print format_acrs_none % (scope, model_id, training_days)
                else:
                    median_abs_error = median(abs_errors)
                    median_rel_error = median(rel_errors)
                    if verbose:
                        print format_acrs_some % (scope, model_id, training_days,
                                                  median_abs_error, median_rel_error)
                    reduced_df.append([scope, model_id, training_days, median_rel_error, median_abs_error])

    # report on the best training_days for each model id for global scope
    pdb.set_trace()
    reduced = reduced_df.result()
    print reduced.shape
    format_header = '%6s %11s %3s %6s %6s'
    format_detail = '%6s %11s %3d %6.0f %6.2f'
    report.append(format_header % ('scope', 'model', 'td', 'mError', 'mRelEr'))
    for model_id in pd.Series.unique(reduced.model_id):
        best_training_days = None
        lowest_median_abs_error = 1e308
        selected_median_rel_error = None
        for training_days in sorted(pd.Series.unique(reduced[reduced.model_id == model_id].training_days)):
            df2 = reduced[(reduced.model_id == model_id) &
                          (reduced.training_days == training_days)]
            assert(len(df2) == 1)
            median_abs_error = df2.median_abs_error.values[0]
            if median_abs_error < lowest_median_abs_error:
                lowest_median_abs_error = median_abs_error
                best_training_days = int(training_days)
                selected_median_rel_error = df2.median_rel_error.values[0]
                #  print 'updated', model_id, best_training_days, lowest_median_abs_error

        report.append(format_detail % (
            'global', model_id, best_training_days, lowest_median_abs_error, selected_median_rel_error))
    pdb.set_trace()


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_log)
    print control

    report = Report()
    for in_file_date in control.in_file_dates:
        in_path = control.path_in_base % in_file_date
        f = open(in_path, 'rb')
        df = pickle.load(f)
        f.close()

        analyze(in_file_date, df, report, control)

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
