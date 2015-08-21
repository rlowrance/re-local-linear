'''create WORKING/chart-06.txt file

INPUT FILES
 WORKING/ege_week-YYYY-MM-DD-df[-test].pickle
 WORKING/ege_week-YYYY-MM-DD-dict[-test].pickle

OUTPUT FILES
 WORKING/chart-06[-test].txt
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
    print 'usage: python chart-05.py YYYY-MM-DD [--test]'
    print ' YYYY-MM-DD centered date for the test transactions'
    print ' --test     if present, read test versions of the input files'


def make_control(argv):
    # return a Bunch

    if len(argv) not in (2, 3):
        print_help()
        sys.exit(1)

    random.seed(123456)
    base_name = argv[0].split('.')[0]
    now = datetime.datetime.now()
    sale_date = argv[1]
    test = parse_command_line.has_arg(argv, '--test')

    b = Bunch(debugging=False,
              path_log=directory('log') + base_name + '.' + now.isoformat('T') + '.log',
              path_in=directory('working') + ('ege_summary_by_scope-%s.pickle' % sale_date),
              path_in_df='%s%s-%s-df%s.pickle' % (
                  directory('working'),
                  'ege_week',
                  sale_date,
                  '-test' if test else ''),
              path_in_dict='%s%s-%s-dict%s.pickle' % (
                  directory('working'),
                  'ege_week',
                  sale_date,
                  '-test' if test else ''),
              path_out='%s%s%s.txt' % (
                  directory('working'),
                  base_name,
                  '-test' if test else ''),
              sale_date=sale_date,
              test=test)

    return b


def median(lst):
    return np.median(np.array(lst, dtype=np.float64))


def analyze(df, all_results, control):
    'create Report showing performance of each model in training week and next week'

    def make_training_days(all_results):
        'return iterable'
        training_days = set()
        for k in all_results.keys():
            training_days.add(k[2])
        return sorted(list(training_days))

    def make_model_names(all_results):
        'return iterable of pairs: (model name, model variant)'
        model_names = set()
        for k in all_results.keys():
            model_names.add(k[3])
        return sorted(list(model_names))

    def make_variants(all_results, model_name):
        variants = set()
        for k, v in all_results.iteritems():
            if k[3] != model_name:
                continue
            for kk, vv in v.iteritems():
                variants.add(kk)
        return sorted(list(variants))

    def make_scopes(all_results):
        'return iterable'
        raise RuntimeError('implement me')

    def get_result(all_results, fold_number, training_days, model, variant, scope):
        'return train and test absolute and relative errors'
        yyyy, mm, dd = control.sale_date.split('-')
        sale_date = datetime.date(int(yyyy), int(mm), int(dd))
        key = (fold_number, sale_date, training_days, model, scope)
        variant_results = all_results[key]
        result = variant_results[variant]
        return result

    def errors(result):
        def abs_rel(actual, estimate):
            abs_error = np.abs(actual - estimate)
            rel_error = abs_error / actual
            return np.median(abs_error), np.median(rel_error)

        def ae(suffix):
            return abs_rel(result['actuals' + suffix], result['estimates' + suffix])

        return ae(''), ae('_next')

    def make_model_id(model, variant):
        if model == 'ols':
            return '%3s %3s %3s' % (model, variant[1][:3], variant[3][:3])
        elif model == 'rf':
            return '%2s %4d    ' % (model, variant[1])
        else:
            print model, variant
            raise RuntimeError('model: ' + str(model))

    pdb.set_trace()
    report = Report()
    report.append('Chart 06')
    report.append('sale_date: %s' % control.sale_date)

    format_header = '%s %3s %11s %6s %7s %7s %7s %7s'
    format_detail = '%d %3d %11s %6s %7.0f %7.2f %7.0f %7.2f'

    report.append(format_header % (
        ' ', ' ', ' ', ' ', 'now', 'now', 'next', 'next'))
    report.append(format_header % (
        'f', 'td', 'model_id', 'scope', 'med abs', 'med rel', 'med abs', 'med rel'))

    for fold_number in xrange(10):
        for training_days in make_training_days(all_results):
            for model in make_model_names(all_results):
                for variant in make_variants(all_results, model):
                    for scope in ('global',) if control.test else make_scopes(df):
                        result = get_result(all_results, fold_number, training_days, model, variant, scope)
                        e = errors(result)
                        now_mae, now_mre = e[0]
                        next_mae, next_mre = e[1]
                        report.append(format_detail % (
                            fold_number, training_days, make_model_id(model, variant),
                            scope,
                            now_mae, now_mre,
                            next_mae, next_mre))
    pdb.set_trace()
    return report


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_log)
    print control

    f = open(control.path_in_df, 'rb')
    df = pickle.load(f)
    f.close()

    f = open(control.path_in_dict, 'rb')
    loaded = pickle.load(f)
    all_results = loaded['all_results']
    f.close

    report = analyze(df, all_results, control)
    pdb.set_trace()
    report.write(control.path_out)

    print control
    if control.test:
        print 'DISCARD OUTPUT: test'
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
