'''create WORKING/chart-05-X.txt files

INPUT FILES: WORKING/ege_summary_by_scope-YYYY-MM-DD.pickle
OUTPUT FILES: WORKING/chart-05.txt
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
    print 'python chart-05.py YYYY-MM-DD'


def make_control(argv):
    # return a Bunch

    random.seed(123456)
    base_name = argv[0].split('.')[0]
    now = datetime.datetime.now()
    sale_date = argv[1]
    testing = False

    b = Bunch(debugging=False,
              path_log=directory('log') + base_name + '.' + now.isoformat('T') + '.log',
              path_in=directory('working') + ('ege_summary_by_scope-%s.pickle' % sale_date),
              path_out=directory('working') + base_name + '.txt',
              sale_date=sale_date,
              testing=testing)

    return b


def median(lst):
    return np.median(np.array(lst, dtype=np.float64))


def analyze(df, control):
    'report best training days for each model'

    def unique(column_name):
        return pd.Series.unique(df[column_name])

    def best_training_days(scope, model_id):
        verbose = False
        lowest_median_abs_error = 1e308
        selected_training_days = None
        selected_median_rel_error = None
        selected_folds = None
        for training_days in unique('training_days'):
            df2 = df[(df.scope == scope) &
                     (df.model_id == model_id) &
                     (df.training_days == training_days)]
            if len(df2) == 0:
                if verbose:
                    print 'no data', scope, model_id, int(training_days), len(df2)
                continue
            assert(len(df2) == 1)

            def value(column_name):
                return df2[column_name].values[0]

            median_abs_error = value('median_abs_error')
            if median_abs_error < lowest_median_abs_error:
                lowest_median_abs_error = median_abs_error
                selected_training_days = int(training_days)
                selected_median_rel_error = value('median_rel_error')
                selected_folds = value('num_folds')
                #  print 'updated', model_id, best_training_days, lowest_median_abs_error

        return selected_training_days, lowest_median_abs_error, selected_median_rel_error, selected_folds

    report = Report()
    report.append('sale_date: %s' % control.sale_date)

    format_header = '%6s %11s %5s %6s %6s %4s'
    format_detail = '%6s %11s %5d %6.0f %6.2f %4d'

    report.append(format_header % (' ', ' ', 'train', 'median', 'median', ' '))
    report.append(format_header % ('scope', 'model', 'days', 'Abs Er', 'Rel Er', 'folds'))

    for scope in unique('scope'):
        for model_id in unique('model_id'):
            td, abs_error, rel_error, folds = best_training_days(scope, model_id)
            report.append(format_detail % (scope, model_id, td, abs_error, rel_error, folds))

    return report


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_log)
    print control

    f = open(control.path_in, 'rb')
    df = pickle.load(f)
    f.close()

    report = analyze(df, control)
    report.write(control.path_out)

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
