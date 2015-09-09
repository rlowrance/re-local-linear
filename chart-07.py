'''create WORKING/chart-07-*.txt files

INPUT FILES
 WORKING/ege_week/YYYY-MM-DD-MODEL-TD-HPs.pickle containing a dict

OUTPUT FILES
 WORKING/chart-07-model-scope-td-ci*.txt
'''

import collections
import cPickle as pickle
import datetime
import numpy as np
import operator
import os
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


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage: python chart-05.py YYYY-MM-DD [--test]'
    print ' YYYY-MM-DD centered date for the test transactions'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    if len(argv) not in (2, 3):
        usage()

    random.seed(123456)

    base_name = argv[0].split('.')[0]
    now = datetime.datetime.now()
    sale_date = argv[1]
    sale_date_split = sale_date.split('-')

    test = parse_command_line.has_arg(argv, '--test')
    debug = False

    def make_path_out(*report_names):
        def make_path(report_name):
            return '%s%s-%s.txt' % (directory('working'), base_name, report_name)

        d = {report_name: make_path(report_name) for report_name in report_names}
        return d

    return Bunch(
        debug=debug,
        test=test,
        path_log=directory('log') + base_name + '.' + now.isoformat('T') + '.log',
        path_in=directory('working') + ('ege_summary_by_scope-%s.pickle' % sale_date),
        dir_in=directory('working') + 'ege_week/' + argv[1] + '/',
        path_out=make_path_out('model_scope_td_ci_abserror',
                               'model_scope_td_ci_relerror',
                               'model_scope_td_ci_abserror_reduced',
                               'drift_all_models',
                               'drift_best_models'),
        sale_date=sale_date,
        sale_year=sale_date_split[0],
        sale_month=sale_date_split[1],
        sale_day=sale_date_split[2],
        ci_n_samples=10000,  # samples to stochastically estimate confidence intervals
        ci_low=2.5,
        ci_high=97.5,
        n_folds=10,
        now=str(now),
    )


def median(lst):
    return np.median(np.array(lst, dtype=np.float64))


Key = collections.namedtuple('Key', 'model training_days, variant, fold_number')


def analyze(all_results, control):
    'create Report showing performance of each model in training week and next week'

    def make_scopes():
        return ('global',)  # for now; later we plan knn-based scopes

    def get_result(all_results, fold_number, training_days, model, variant):
        'return train and test absolute and relative errors'
        key = Key(fold_number=fold_number,
                  training_days=training_days,
                  model=model,
                  variant=variant)
        return all_results[key]

    def make_model_id(model, variant):
        if model == 'lr':
            return '%2s %3s %3s' % (model, variant[1][:3], variant[3][:3])
        elif model == 'rf':
            return '%2s %2d' % (model, variant[1])
        else:
            assert False, (model, variant)

    def us(lst):
        'return unique sorted values in lst'
        return sorted(set(lst))

    def natural(x):
        '''if in log units, convert to natural units

        This is a kludgy. An alternative is to examine the variant
        and determine if the y value is in the log domain
        '''
        return np.exp(x) if (np.all((x > 0) & (x < 20))) else x

    def abs_errors(actuals, estimates):
        return np.abs(natural(actuals) - natural(estimates))

    def now_absolute_errors(result):
        'return np.array 1D'
        return abs_errors(result['actuals'], result['estimates'])

    def next_absolute_errors(result):
        'return np.array 1D'
        return abs_errors(result['actuals_next'], result['estimates_next'])

    def rel_errors(actuals, estimates):
        'relative absolute errors'
        rel_errors = abs_errors(actuals, estimates) / natural(actuals)
        return rel_errors

    def now_relative_errors(result):
        return rel_errors(result['actuals'], result['estimates'])

    def next_relative_errors(result):
        return rel_errors(result['actuals_next'], result['estimates_next'])

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

    # report on drifts
    def drifts_report(details, id):
        'return Report() object'
        drifts = Report()
        drifts.append('Chart 07: Summary of Drifts')
        drifts.append(control.now)
        if control.test:
            drifts.append('DISCARD: TESTING')
        drifts.append('For ' + id)
        drifts.append('Drift := Ratio of median error in next period to current period')
        drifts_header = '%9s %9s'
        drifts_detail = '%9s %9.2f'
        drifts.append(drifts_header % ('statistic', 'value'))
        all_drifts_np = np.array(all_abs_drifts)
        drifts.append(drifts_detail % ('min', np.min(all_drifts_np)))
        drifts.append(drifts_detail % ('max', np.max(all_drifts_np)))
        drifts.append(drifts_detail % ('mean', np.mean(all_drifts_np)))
        drifts.append(drifts_detail % ('median', np.median(all_drifts_np)))
        return drifts

    # reports setup
    pdb.set_trace()
    model_scope_td_ci_abserror = Report()
    model_scope_td_ci_relerror = Report()
    model_scope_td_ci_abserror_reduced = Report()

    def append_common(line):
        model_scope_td_ci_abserror.append(line)
        model_scope_td_ci_abserror_reduced.append(line)
        model_scope_td_ci_relerror.append(line)

    append_common('Chart 07: 95% Confidence Intervals')
    append_common('Summarizing Across the Cross Validation Folds')
    model_scope_td_ci_abserror.append('Absolute Errors in Dollars')
    model_scope_td_ci_relerror.append('Relative Absolute Errors')
    append_common(control.now)
    model_scope_td_ci_abserror_reduced.append('Reduced to Lowest Median Now and Next Errors')
    if control.test:
        append_common('TESTING: DISCARD')

    format_header1 = '%10s %3s %23s %4s %23s %4s %5s'
    format_header2 = '%10s %3s %7s %7s %7s %4s %7s %7s %7s %4s %5s'
    format_detail_abs = '%-10s %3s %7.0f %7.0f %7.0f %4d %7.0f %7.0f %7.0f %4d %5.3f'
    format_detail_rel = '%-10s %3s %7.3f %7.3f %7.3f %4d %7.3f %7.3f %7.3f %4d %5.3f'

    assert control.ci_high - control.ci_low == 95.0, 'change header if you change the ci'

    def print_header():
        def center(s, width):
            def pad(extra, s):
                return (s if extra <= 0 else
                        s + ' ' if extra == 1 else
                        pad(extra - 2, ' ' + s + ' '))

            return pad(max(0, width - len(s)), s)

        append_common('Scope: ' + scope)
        append_common(' ')
        append_common(format_header1 % (
            '', '', center('now 95 pct ci', 23), '', center('next 95 pct ci', 23), '', ''))
        append_common(format_header2 % (
            'model_id', 'td', 'low', 'median', 'high', 'n', 'low', 'median', 'high', 'n', 'drift'))

    details_abs = []  # detail line info
    details_rel = []
    all_abs_drifts = []
    all_rel_drifts = []
    keys = all_results.keys()
    for scope in make_scopes():
        assert scope == 'global', scope
        print_header()
        for model in us([k.model
                         for k in keys]):
            for variant in us([k.variant
                               for k in keys
                               if k.model == model]):
                for training_days in us([k.training_days
                                         for k in keys
                                         if k.model == model
                                         if k.variant == variant]):
                    now_accumulated_abs_errors = []
                    next_accumulated_abs_errors = []
                    now_accumulated_rel_errors = []
                    next_accumulated_rel_errors = []
                    for fold_number in us([k.fold_number
                                           for k in keys
                                           if k.model == model
                                           if k.variant == variant
                                           if k.training_days == training_days]):
                        if False and model == 'lr' and variant[1] == 'log' and variant[3] == 'linear' and training_days == 7:
                            pdb.set_trace()
                        result = get_result(all_results, fold_number, training_days, model, variant)
                        now_accumulated_abs_errors.extend(now_absolute_errors(result))
                        next_accumulated_abs_errors.extend(next_absolute_errors(result))
                        now_accumulated_rel_errors.extend(now_relative_errors(result))
                        next_accumulated_rel_errors.extend(next_relative_errors(result))
                    if False and model == 'lr' and variant[1] == 'log' and variant[3] == 'linear' and training_days == 7:
                        pdb.set_trace()
                    now_abs_low, now_abs_high = confidence_interval(now_accumulated_abs_errors,
                                                                    control.ci_low,
                                                                    control.ci_high,
                                                                    control.ci_n_samples)
                    next_abs_low, next_abs_high = confidence_interval(next_accumulated_abs_errors,
                                                                      control.ci_low,
                                                                      control.ci_high,
                                                                      control.ci_n_samples)
                    now_rel_low, now_rel_high = confidence_interval(now_accumulated_rel_errors,
                                                                    control.ci_low,
                                                                    control.ci_high,
                                                                    control.ci_n_samples)
                    next_rel_low, next_rel_high = confidence_interval(next_accumulated_rel_errors,
                                                                      control.ci_low,
                                                                      control.ci_high,
                                                                      control.ci_n_samples)
                    model_id = make_model_id(model, variant)
                    median_abs_now = np.median(now_accumulated_abs_errors)
                    median_abs_next = np.median(next_accumulated_abs_errors)
                    median_rel_now = np.median(now_accumulated_rel_errors)
                    median_rel_next = np.median(next_accumulated_rel_errors)
                    abs_drift = median_abs_next / median_abs_now
                    all_abs_drifts.append(abs_drift)
                    rel_drift = median_rel_next / median_rel_now
                    all_rel_drifts.append(rel_drift)
                    detail_abs = (
                        model_id, training_days,
                        now_abs_low, median_abs_now, now_abs_high, len(now_accumulated_abs_errors),
                        next_abs_low, median_abs_next, next_abs_high, len(next_accumulated_abs_errors),
                        abs_drift,
                    )
                    details_abs.append(detail_abs)
                    model_scope_td_ci_abserror.append(format_detail_abs % detail_abs)
                    detail_rel = (
                        model_id, training_days,
                        now_rel_low, median_rel_now, now_rel_high, len(now_accumulated_rel_errors),
                        next_rel_low, median_rel_next, next_rel_high, len(next_accumulated_rel_errors),
                        rel_drift,
                    )
                    details_rel.append(detail_abs)
                    model_scope_td_ci_relerror.append(format_detail_rel % detail_rel)

    drift_all_models = drifts_report(all_abs_drifts, 'All Models')

    # reduce to best now and next lines
    pdb.set_trace()
    format_regret = '   %6s %6.2f'
    reduction_details = []
    regrets = []
    for model_id in sorted(set([d[0] for d in details_abs])):
        # examine the detail lines for the model_id
        best_now = 1e308
        best_next = 1e308
        detail_now = None
        detail_next = None
        for detail in [d for d in details_abs if d[0] == model_id]:
            if detail[3] < best_now:
                best_now = detail[3]
                detail_now = detail
            if detail[7] < best_next:
                best_next = detail[7]
                detail_next = detail
        # print the best now and next model details
        regret = detail_now[7] / detail_next[7]
        regrets.append(regret)
        model_scope_td_ci_abserror_reduced.append(format_detail_abs % detail_now)
        model_scope_td_ci_abserror_reduced.append(format_detail_abs % detail_next)
        model_scope_td_ci_abserror_reduced.append(format_regret % ('regret', regret))
        reduction_details.append(detail_now)
        reduction_details.append(detail_next)

    model_scope_td_ci_abserror_reduced.append(' ')
    model_scope_td_ci_abserror_reduced.append('Statistics on the regret')
    x = np.array(regrets)
    model_scope_td_ci_abserror_reduced.append(format_regret % ('min', np.min(x)))
    model_scope_td_ci_abserror_reduced.append(format_regret % ('max', np.max(x)))
    model_scope_td_ci_abserror_reduced.append(format_regret % ('mean', np.mean(x)))
    model_scope_td_ci_abserror_reduced.append(format_regret % ('median', np.median(x)))

    pdb.set_trace()
    drift_best_models = drifts_report(reduction_details, 'Best Models')

    # TODO: report on regret

    return {'model_scope_td_ci_abserror': model_scope_td_ci_abserror,
            'model_scope_td_ci_relerror': model_scope_td_ci_relerror,
            'model_scope_td_ci_abserror_reduced': model_scope_td_ci_abserror_reduced,
            'drift_all_models': drift_all_models,
            'drift_best_models': drift_best_models}


def read_all_results(control):
    def same_date(base_name):
        split = base_name.split('-')
        return (split[0] == control.sale_year and
                split[1] == control.sale_month and
                split[2] == control.sale_day)

    def file_with_fold_number(base_name, fold_number):
        return '%s-%d.pickle' % (base_name, fold_number)

    def all_folds_present(base_name, file_names):
        for fold_number in xrange(control.n_folds):
            if file_with_fold_number(base_name, fold_number) not in file_names:
                return False
        return True

    def get_base_name(file_name):
        splits = file_name.split('-')
        return '-'.join(splits[0:-1])

    def get_model(base_name):
        return file_name.split('-')[3]

    def get_training_days(base_name):
        return file_name.split('-')[4]

    def read_file(model, training_days, model_path, file_name):
        'append new key and value to all_results'
        splits = file_name.split('-')
        fold_number = splits[-1].split('.')[0]
        f = open(model_path + file_name, 'rb')
        pickled = pickle.load(f)
        f.close()
        variant = pickled['variant']
        result = pickled['result']
        key = Key(model, training_days, variant, int(fold_number))
        return key, result

    all_results = {}
    for dir_name in os.listdir(control.dir_in):
        if dir_name[0] == '.':
            continue  # ex: found .DS_Store on Mac OS
        model, training_days = dir_name.split('-')
        model_path = control.dir_in + dir_name + '/'
        for file_name in os.listdir(model_path):
            if file_name[0] == '.':
                continue
            file_key, file_variant = read_file(model, int(training_days), model_path, file_name)
            all_results[file_key] = file_variant
        if control.test:
            if len(all_results) > 100:
                break

    print '# all results', len(all_results)
    return all_results


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_log)
    print control

    all_results = read_all_results(control)

    reports = analyze(all_results, control)
    pdb.set_trace()
    for k, v in reports.iteritems():
        v.write(control.path_out[k])

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
