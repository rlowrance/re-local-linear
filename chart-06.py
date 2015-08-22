'''create WORKING/chart-06.txt file

INPUT FILES
 WORKING/ege_week-YYYY-MM-DD-df[-test].pickle
 WORKING/ege_week-YYYY-MM-DD-dict[-test].pickle

OUTPUT FILES
 WORKING/chart-06-each-fold[-test].txt
 WORKING/chart-06-across-folds[-test].txt
 WORKING/chart-07-best-across-folds[-test].txt
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
              path_out_report_each_fold='%s%s%s%s.txt' % (
                  directory('working'),
                  base_name,
                  '-each-fold',
                  '-test' if test else ''),
              path_out_report_across_folds='%s%s%s%s.txt' % (
                  directory('working'),
                  base_name,
                  '-across-folds',
                  '-test' if test else ''),
              path_out_report_best_across_folds='%s%s%s%s.txt' % (
                  directory('working'),
                  base_name,
                  '-best-across-folds',
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

    def make_fold_numbers(all_results):
        fold_numbers = set()
        for k in all_results.keys():
            fold_numbers.add(k[0])
        return sorted(list(fold_numbers))

    def make_scopes(all_results):
        'return iterable'
        scopes = set()
        for k in all_results.keys():
            scopes.add(k[4])
        return sorted(list(scopes))

    def get_result(all_results, fold_number, training_days, model, variant, scope):
        'return train and test absolute and relative errors'
        yyyy, mm, dd = control.sale_date.split('-')
        sale_date = datetime.date(int(yyyy), int(mm), int(dd))
        key = (fold_number, sale_date, training_days, model, scope)
        variant_results = all_results[key]
        result = variant_results[variant]
        return result

    def abs_rel_errors(actual_raw_list, estimate_raw_list):
        '''return absolute and relative errors, after possibly transforming from log to natural units

        ARGS TYPES
        either python list or np.array
        '''
        assert len(actual_raw_list) == len(estimate_raw_list)
        # convert possibly lists to np.arrays
        actual_raw = np.array(actual_raw_list, dtype=np.float64)
        estimate_raw = np.array(estimate_raw_list, dtype=np.float64)
        # assume small values are in log domain and transform them to natural units
        actual = np.exp(actual_raw) if np.all(actual_raw < 20.0) else actual_raw
        estimate = np.exp(estimate_raw) if np.all(estimate_raw < 20.0) else estimate_raw
        # now values are in natural units stored in np.arrays
        abs_error = np.abs(actual - estimate)
        rel_error = abs_error / actual
        return np.median(abs_error), np.median(rel_error)

    def errors(result):

        def ae(suffix):
            return abs_rel_errors(result['actuals' + suffix], result['estimates' + suffix])

        return ae(''), ae('_next')

    def median_across_folds(accumulated):
        'return medians: mae now, mre now, mae next, mre next'
        actuals_now = []
        estimates_now = []
        actuals_next = []
        estimates_next = []
        for result in accumulated:
            actuals_now.extend(result['actuals'])
            estimates_now.extend(result['estimates'])
            actuals_next.extend(result['actuals_next'])
            estimates_next.extend(result['estimates_next'])
        errors_now = abs_rel_errors(actuals_now, estimates_now)
        errors_next = abs_rel_errors(actuals_next, estimates_next)
        return errors_now[0], errors_now[1], errors_next[0], errors_next[1]

    def make_model_id(model, variant):
        if model == 'ols':
            return '%3s %3s %3s' % (model, variant[1][:3], variant[3][:3])
        elif model == 'rf':
            return '%2s %4d    ' % (model, variant[1])
        else:
            print model, variant
            raise RuntimeError('model: ' + str(model))

    def print_best_results(report, across):
        'append lines with lowest now and next errors'
        now_best = 1e307
        next_best = 1e307
        for k, v in across.iteritems():
            now_mae, next_mae, line = v
            if now_mae < now_best:
                now_best = now_mae
                now_line = line
            if next_mae < next_best:
                next_best = next_mae
                next_line = line
        report_best_across_folds.append(now_line)
        report_best_across_folds.append(next_line)

    report_each_fold = Report()
    report_across_folds = Report()
    report_best_across_folds = Report()

    report_each_fold.append('Chart 06: Accuracy by Fold')
    report_across_folds.append('Chart 06: Accuracy Across Folds')
    report_best_across_folds.append('Chart 06: Best model_id, scope, td Across Folds')

    def append_12(line):
        report_each_fold.append(line)
        report_across_folds.append(line)

    def append_123(line):
        append_12(line)
        report_best_across_folds.append(line)

    append_123('sale_date: %s' % control.sale_date)

    folds_format_header = '%11s %6s %3s %1s %7s %7s %7s %7s'
    folds_format_detail = '%11s %6s %3d %1s %7.0f %7.2f %7.0f %7.2f'

    append_123(folds_format_header % (
        ' ', ' ', ' ', ' ', 'now', 'now', 'next', 'next'))
    append_123(folds_format_header % (
        'model_id', 'scope', 'td', 'f', 'med abs', 'med rel', 'med abs', 'med rel'))

    pdb.set_trace()
    for model in make_model_names(all_results):
        for variant in make_variants(all_results, model):
            for scope in make_scopes(all_results):
                across = {}
                if control.test and scope != 'global':
                    continue
                for training_days in make_training_days(all_results):
                    accumulated_results = []
                    for fold_number in make_fold_numbers(all_results):
                        result = get_result(all_results, fold_number, training_days, model, variant, scope)
                        accumulated_results.append(result)
                        e = errors(result)
                        now_mae, now_mre = e[0]
                        next_mae, next_mre = e[1]
                        report_each_fold.append(folds_format_detail % (
                            make_model_id(model, variant),
                            scope,
                            training_days,
                            str(fold_number),
                            now_mae,
                            now_mre,
                            next_mae,
                            next_mre))
                    # write summary line across folds
                    now_mae, now_mre, next_mae, next_mre = median_across_folds(accumulated_results)
                    model_id = make_model_id(model, variant)
                    line = folds_format_detail % (
                        model_id,
                        scope,
                        training_days,
                        'M',  # median across folds
                        now_mae,
                        now_mre,
                        next_mae,
                        next_mre)
                    append_12(line)
                    across[training_days] = (now_mae, next_mae, line)
            # finished model, variant, scope
            # find best models by examining across
            print_best_results(report_across_folds, across)
    pdb.set_trace()

    return report_each_fold, report_across_folds, report_best_across_folds


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

    report_each_fold, report_across_folds, report_best_across_folds = analyze(df, all_results, control)
    report_each_fold.write(control.path_out_report_each_fold)
    report_across_folds.write(control.path_out_report_across_folds)
    report_best_across_folds.write(control.path_out_report_best_across_folds)

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
