'''create WORKING/chart-07-*.txt files

INPUT FILES
 WORKING/ege_week-YYYY-MM-DD-dict[-test].pickle

OUTPUT FILES
 WORKING/chart-07-each-fold[-test].txt
 WORKING/chart-07-across-folds[-test].txt
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
from ege_week import Rf, Lr


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

    return Bunch(
        debugging=False,
        path_log=directory('log') + base_name + '.' + now.isoformat('T') + '.log',
        path_in=directory('working') + ('ege_summary_by_scope-%s.pickle' % sale_date),
        path_in_dict='%s%s-%s-dict%s.pickle' % (
            directory('working'),
            'ege_week',
            sale_date,
            '-test' if test else ''),
        path_out_report_model_scope_td_ci='%s%s%s%s.txt' % (
            directory('working'),
            base_name,
            '-model-scope-td-ci',
            '-test' if test else ''),
        sale_date=sale_date,
        test=test,
        ci_n_samples=10000,  # samples to stochastically estimate confidence intervals
        ci_low=2.5,
        ci_high=97.5,
    )


def median(lst):
    return np.median(np.array(lst, dtype=np.float64))


def analyze(all_results, control):
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

    def make_model_id(model, variant):
        if model == 'lr':
            return '%2s %3s %3s' % (model, variant[1][:3], variant[3][:3])
        elif model == 'rf':
            return '%2s %2d   ' % (model, variant[1])
        else:
            assert False, (model, variant)

    def absolute_errors(result):
        'return np.array 1D'
        actuals = result['actuals']
        estimates = result['estimates']
        return np.abs(actuals - estimates)

    def confidence_interval(samples, low_percentile, high_percentile, n_samples):
        '''return a and b such that [a,b] is a 95% confidence interval for the samples

        ARGS
        low_percentile, high_percentile: in [0,100]
        '''
        assert low_percentile >= 0 and low_percentile <= 100
        assert high_percentile >= 0 and high_percentile <= 100
        resamples = np.random.choice(samples, size=n_samples, replace=True)
        low = np.percentile(resamples, low_percentile)
        high = np.percentile(resamples, high_percentile)
        return low, high

    # reports setup
    model_scope_td_ci = Report()
    model_scope_td_ci.append('Chart 07: 95% Confidence Intervals')

    format_header = '%10s %6s %3s %23s'
    format_detail = '%10s %6s %3s %7.0f %7.0f %7.0f'

    assert control.ci_high - control.ci_low == 95.0
    model_scope_td_ci.append(format_header % ('', '', '', '[  95 pct ci   ]'))
    model_scope_td_ci.append(format_header % ('model_id', 'scope', 'td', '   low median  high'))

    pdb.set_trace()
    for model in make_model_names(all_results):
        for variant in make_variants(all_results, model):
            for scope in make_scopes(all_results):
                if control.test and scope != 'global':
                    # skip zip codes if testing
                    continue
                for training_days in make_training_days(all_results):
                    accumulated_errors = []
                    for fold_number in make_fold_numbers(all_results):
                        result = get_result(all_results, fold_number, training_days, model, variant, scope)
                        accumulated_errors.extend(absolute_errors(result))
                    low, high = confidence_interval(accumulated_errors,
                                                    control.ci_low,
                                                    control.ci_high,
                                                    control.ci_n_samples)
                    model_scope_td_ci.append(format_detail % (
                        make_model_id(model, variant), scope, training_days,
                        low, np.median(accumulated_errors), high))
    return model_scope_td_ci


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_log)
    print control

#    f = open(control.path_in_df, 'rb')
#    df = pickle.load(f)
#    f.close()

    f = open(control.path_in_dict, 'rb')
    loaded = pickle.load(f)
    all_results = loaded['all_results']
    f.close

    report1 = analyze(all_results, control)
    report1.write(control.path_out_report_model_scope_td_ci)

    print control
    if control.test:
        print 'DISCARD OUTPUT: test'
    print 'done'

    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        Rf()
        Lr()
        pprint()
        parse_command_line()
        pd.DataFrame()
        np.array()
        operator.add()

    main(sys.argv)
