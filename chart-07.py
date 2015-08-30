'''create WORKING/chart-07-*.txt files

INPUT FILES
 WORKING/ege_week/YYYY-MM-DD-MODEL-TD-HPs.pickle containing a dict

OUTPUT FILES
 WORKING/chart-07-model-scope-td-ci.txt
'''

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

    test = True
    debug = False

    return Bunch(
        debug=debug,
        test=test,
        path_log=directory('log') + base_name + '.' + now.isoformat('T') + '.log',
        path_in=directory('working') + ('ege_summary_by_scope-%s.pickle' % sale_date),
        dir_in=directory('working') + 'ege_week/',
        path_out_report_model_scope_td_ci='%s%s%s%s.txt' % (
            directory('working'),
            base_name,
            '-model-scope-td-ci',
            '-test' if test else ''),
        sale_date=sale_date,
        sale_year=sale_date_split[0],
        sale_month=sale_date_split[1],
        sale_day=sale_date_split[2],
        ci_n_samples=10000,  # samples to stochastically estimate confidence intervals
        ci_low=2.5,
        ci_high=97.5,
        n_folds=10,
    )


def median(lst):
    return np.median(np.array(lst, dtype=np.float64))


class Key(object):
    'key to all_results dict'
    def __init__(self, model, training_days, variant, fold_number):
        self.model = model
        self.training_days = training_days
        self.variant = variant
        self.fold_number = fold_number

    def get_model(self):
        return self.model

    def get_training_days(self):
        return self.training_days

    def get_variant(self):
        return self.variant

    def get_fold_number(self):
        return self.fold_number

    def __repr__(self):
        return 'Key(%s, %s, %s, %s)' % (
            self.model, str(self.variant), str(self.training_days), str(self.fold_number))

    def _get_key(self):
        return (self.model, self.training_days, self.variant, self.fold_number)

    def __hash__(self):
        return hash(self._get_key())

    def __eq__(self, other):
        return (self.model == other.model and
                self.training_days == other.training_days and
                self.variant == other.variant and
                self.fold_number == other.fold_number)


def analyze(all_results, control):
    'create Report showing performance of each model in training week and next week'

    def make_training_days():
        return sorted([k.get_training_days() for k in all_results.keys()])

    def make_model_names():
        return sorted([k.get_model() for k in all_results.keys()])

    def make_fold_numbers():
        return sorted([k.get_fold_number() for k in all_results.keys()])

    def make_variants(model_name):
        return sorted([k.get_variant() for k in all_results.keys()])

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

    def summarize_drifts(all_drifts, report):
        x = np.array(all_drifts)
        report.append(' ')
        report.append('Summary of Drifts')
        report.append('min:    ' + str(np.min(x)))
        report.append('max:    ' + str(np.max(x)))
        report.append('mean:   ' + str(np.mean(x)))
        report.append('median: ' + str(np.median(x)))

    # reports setup
    pdb.set_trace()
    model_scope_td_ci = Report()
    model_scope_td_ci.append('Chart 07: 95% Confidence Intervals')
    model_scope_td_ci.append('Summarizing Across the Cross Validation Folds')

    format_header1 = '%10s %3s %23s %4s %23s %4s %5s'
    format_header2 = '%10s %3s %7s %7s %7s %4s %7s %7s %7s %4s %5s'
    format_detail = '%-10s %3s %7.0f %7.0f %7.0f %4d %7.0f %7.0f %7.0f %4d %5.3f'

    assert control.ci_high - control.ci_low == 95.0, 'change header if you change the ci'

    def print_header():
        model_scope_td_ci.append('Scope: ' + scope)
        model_scope_td_ci.append(' ')
        pct_ci = '[       95 pct ci     ]'
        model_scope_td_ci.append(format_header1 % ('', '', pct_ci, '', pct_ci, '', ''))
        model_scope_td_ci.append(format_header2 % (
            'model_id', 'td', 'low', 'median', 'high', 'n', 'low', 'median', 'high', 'n', 'drift'))

    lines = []  # detail lines
    all_drifts = []
    keys = all_results.keys()
    for scope in make_scopes():
        assert scope == 'global', scope
        print_header()
        for model in us([k.get_model()
                         for k in keys]):
            for variant in us([k.get_variant()
                               for k in keys
                               if k.get_model() == model]):
                for training_days in us([k.get_training_days()
                                         for k in keys
                                         if k.get_model() == model
                                         if k.get_variant() == variant]):
                    now_accumulated_errors = []
                    next_accumulated_errors = []
                    for fold_number in us([k.get_fold_number()
                                           for k in keys
                                           if k.get_model() == model
                                           if k.get_variant() == variant
                                           if k.get_training_days() == training_days]):
                        result = get_result(all_results, fold_number, training_days, model, variant)
                        now_accumulated_errors.extend(now_absolute_errors(result))
                        next_accumulated_errors.extend(next_absolute_errors(result))
                    now_low, now_high = confidence_interval(now_accumulated_errors,
                                                            control.ci_low,
                                                            control.ci_high,
                                                            control.ci_n_samples)
                    next_low, next_high = confidence_interval(next_accumulated_errors,
                                                              control.ci_low,
                                                              control.ci_high,
                                                              control.ci_n_samples)
                    model_id = make_model_id(model, variant)
                    median_now = np.median(now_accumulated_errors)
                    median_next = np.median(next_accumulated_errors)
                    drift = median_next / median_now
                    all_drifts.append(drift)
                    line = format_detail % (
                        model_id, training_days,
                        now_low, median_now, now_high, len(now_accumulated_errors),
                        next_low, median_next, next_high, len(next_accumulated_errors),
                        drift,
                    )
                    model_scope_td_ci.append(line)
                    lines.append((model_id, training_days, median_now, median_next, line))

    summarize_drifts(all_drifts, model_scope_td_ci)

    # TODO: produce reduced report using info in lines

    return model_scope_td_ci


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

    processed = set()
    all_results = {}
    file_names = set(os.listdir(control.dir_in))
    for file_name in file_names:
        if control.test and len(all_results) > 1000:
            break
        base_name = get_base_name(file_name)  # drop suffix -fold.pickle
        if base_name not in processed:
            if same_date(base_name):
                if all_folds_present(base_name, file_names):
                    # accumulate into all_results
                    for fold_number in xrange(0, control.n_folds):
                        f = open(control.dir_in + file_with_fold_number(base_name, fold_number), 'rb')
                        pickled = pickle.load(f)
                        f.close()
                        variant = pickled['variant']
                        result = pickled['result']
                        key = Key(model=get_model(base_name),
                                  training_days=int(get_training_days(base_name)),
                                  variant=variant,
                                  fold_number=fold_number)
                        all_results[key] = result
            processed.add(base_name)
    print '# all_results', len(all_results)
    return all_results


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_log)
    print control

    all_results = read_all_results(control)
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
