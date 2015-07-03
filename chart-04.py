# create files for chart-04: lassocv (lasso with cross validation)
# invocation python chart-04.py --in PATH [--cache] --units UNITS

# input files:
#  PATH, if --cache is omitted or PATH is new
#  WORKING/chart-04-cache.pickle, if --cache is specified for a prior PATH
#  reduction of the cvcell, if --cache is specified
# output files: a series of charts in c1, c2, ... in  txt files
#  WORKING/chart-04.cache.pickle, if --cache is specified for a new PATH
#  WORKING/chart-04.UNITS-c1.txt
#  WORKING/chart-04.UNITS-c2.txt
#  ...

import collections
import cPickle as pickle
import datetime
import operator
import pdb
import sys

# import my stuff
from Bunch import Bunch
from directory import directory
from Logger import Logger
import parse_command_line


# prevent warning from pyflakes by using pdb
if False:
    pdb.set_trace()


def print_help():
    print 'python chart-04.py --in PATH [--cache] --unit UNITS'
    print 'where'
    print ' PATH is the path to the input file'
    print ' UNITS is one of {natural,rescaled}, used only in chart headings'
    print ' --cache reads from the cache instead of PATH'
    print '     the cache contains a reduction of the input'


def make_control(argv):
    # return a Bunch

    # supply common conrol values
    b = Bunch(debugging=False,
              base_name=argv[0].split('.')[0],
              me=argv[0],
              arg_cache=parse_command_line.has_arg(argv, '--cache'),
              arg_in=parse_command_line.get_arg(argv, '--in'),
              arg_units=parse_command_line.get_arg(argv, '--units'),
              now=datetime.datetime.now(),
              testing=False)

    return b


def cache_path(control):
    # return path to cache
    # in principle, incorporate all the command line parameters into the path
    base = control.base_name + '.cache.' + control.log_file_middle
    suffix = '.pickle'
    return directory('working') + base + suffix


def create_reduction(control):
    '''return reduction of the data in path control.arg_in

    The reduced data is a dict
    key = ERROR (from command line) (one of mrmse mae)
    value = a dicionary with the estimated generalization error
     key =(response, predictor, training_days)
     value = scalar value from each fold
    '''

    def get_cv_result(file_path):
        '''Return CvResult instance.'''
        f = open(file_path, 'rb')
        cv_result = pickle.load(f)
        f.close()
        return cv_result

    # create table containing results from each cross validation
    def cv_result_summary(cv_result):
        if control.specs.metric == 'median-median':
            maybe_value = cv_result.median_of_root_median_squared_errors()
        elif control.specs.metric == 'mean-wi10':
            maybe_value = cv_result.mean_of_fraction_wi10()
        elif control.specs.metric == 'mean-mean':
            maybe_value = cv_result.mean_of_root_mean_squared_errors()
        else:
            print control.specs
            raise RuntimeError('unknown metric: ' + control.specs.metric)
        return maybe_value.value if maybe_value.has_value else None

    # key = (fold_number, sale_date) value = num test transactions
    num_tests = collections.defaultdict(int)
    # key = (fold_number, sale_date, predictor_name) value = coefficient
    test_coef = {}

    path = control.arg_in
    cv_result = get_cv_result(path)  # a CvResult instance
    num_folds = len(cv_result.fold_results)
    for fold_number in xrange(num_folds):
        fold_result = cv_result.fold_results[fold_number]
        # fold_actuals = fold_result.actuals
        # fold_estimates = fold_result.estimates
        fold_raw = fold_result.raw_fold_result
        for sale_date, fitted_model in fold_raw.iteritems():
            # sale_date_num_train = fitted_model['num_train']
            # sale_date_estimates = fitted_model['estimates']
            # sale_date_actuals = fitted_model['estimates']
            sale_date_predictor_names = fitted_model['predictor_names']
            sale_date_num_test = fitted_model['num_test']
            # sale_date_model = fitted_model['model']
            sale_date_fitted = fitted_model['fitted']

            # extract info from the fitted_model
            num_tests[(fold_number, sale_date)] += sale_date_num_test

            # save each coefficient
            sale_date_coef = sale_date_fitted.coef_
            assert(len(sale_date_coef) == len(sale_date_predictor_names))
            for i in xrange(len(sale_date_coef)):
                key = (fold_number, sale_date, sale_date_predictor_names[i])
                test_coef[key] = sale_date_coef[i]

    reduced_data = {'num_tests': num_tests, 'test_coef': test_coef}
    return reduced_data


def create_charts(control, data):
    '''Create all of the txt files from the reduced data'''

    class Report(object):
        def __init__(self):
            self.lines = []

        def append(self, line):
            self.lines.append(line)
            print line

        def extend(self, lines):
            self.lines.extend(lines)
            for line in lines:
                print line

        def write(self, path):
            f = open(path, 'w')
            for line in self.lines:
                f.write(line)
                f.write('\n')
            f.close()

    def make_counts(num_tests, test_coef):
        # ARGS
        #  num_tests[(fold_number, sale_date)] -> count
        #  test_coef[(fold_number, sale_date, predictor_name)] -> coefficient
        # return num_fitted_models, predictors_ordered, Bunch of non_zero_counts

        num_fitted_models = 0  # across all folds
        for fold_number, sale_date in num_tests.iteritems():
            num_fitted_models += 1

        by_predictor = collections.defaultdict(int)
        by_predictor_month = collections.defaultdict(int)
        by_predictor_quarter = collections.defaultdict(int)
        by_predictor_year = collections.defaultdict(int)
        by_year = collections.defaultdict(int)

        month_to_quarter = (1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4)

        for k, coefficient in test_coef.iteritems():
            fold_number, sale_date, predictor = k
            year = sale_date.year
            month = sale_date.month
            quarter = month_to_quarter[month - 1]

            if coefficient != 0:
                by_predictor[predictor] += 1
                by_predictor_month[(predictor, month)] += 1
                by_predictor_quarter[(predictor, quarter)] += 1
                by_predictor_year[(predictor, year)] += 1
                by_year[year] += 1

        # sort predictors in order of popularity
        sorted_by_predictor = \
            sorted(by_predictor.items(),
                   key=operator.itemgetter(1),
                   reverse=True)

        predictors_ordered = []
        for predictor, count in sorted_by_predictor:
            predictors_ordered.append(predictor)

        b = Bunch(by_predictor=by_predictor,
                  by_predictor_month=by_predictor_month,
                  by_predictor_quarter=by_predictor_quarter,
                  by_predictor_year=by_predictor_year,
                  by_year=by_year)

        return num_fitted_models, predictors_ordered, b

    def txt_path(txt_choice):
        prefix = \
            directory('working') + control.base_name + '.' + control.arg_units
        path = prefix + '.' + txt_choice + '.txt'
        return path

    def append_units(report):
        if control.arg_units == 'natural':
            report.append('Features in Natural Units')
        elif control.arg_units == 'rescaled':
            report.append('Features Rescaled to [-1,1]')
        else:
            print 'bad units: ' + control.units
            exit(2)

    def nz_count_all_periods(title,
                             num_fitted_models,
                             predictors_ordered,
                             counts):
        report = Report()
        report.append(title)
        report.append('For All Periods')
        append_units(report)

        format_header = '%25s %8s'
        format_data = '%25s %8.4f'

        report.append(format_header % ('predictor', '%'))
        for predictor in predictors_ordered:

            def percent(predictor):
                numerator = counts.by_predictor[predictor]
                return 100.0 * (numerator * 1.0) / num_fitted_models

            report.append(format_data % (predictor, percent(predictor)))
        report.write(txt_path('nz-count-all-periods'))

    def nz_count_by_year(title, num_fitted_models, predictors_ordered, counts):
        report = Report()
        report.append(title)
        report.append('By Sale Date Year')
        append_units(report)

        num_years = 7  # 2003, 2004, 2005, 2006, 2007, 2008, 2009
        format_header = '%25s' + (' %4d' * num_years) + ' %5s'
        format_data = '%25s' + (' %4.1f' * num_years) + ' %5.1f'

        report.append(format_header % ('predictor name',
                                       2003, 2004, 2005, 2006, 2007, 2008, 2009,
                                       'All'))

        class Percent(object):
            def __init__(self):
                self.total = 0.0

            def percent(self, predictor, year):
                numerator = counts.by_predictor_year[(predictor, year)]
                result = 100.0 * (numerator * 1.0) / num_fitted_models
                self.total += result
                return result

        for predictor in predictors_ordered:

            def percent(self, predictor, year):
                numerator = counts.by_predictor_year[(predictor, year)]
                result = 100.0 * (numerator * 1.0) / num_fitted_models
                self.total += result
                return result

            p = Percent()
            report.append(format_data % (predictor,
                                         p.percent(predictor, 2003),
                                         p.percent(predictor, 2004),
                                         p.percent(predictor, 2005),
                                         p.percent(predictor, 2006),
                                         p.percent(predictor, 2007),
                                         p.percent(predictor, 2008),
                                         p.percent(predictor, 2009),
                                         p.total))
        report.write(txt_path('nz-count-by-year'))

    def ranked_by_year(titles, num_fitted_models, predictors_ordered, counts):
        report = Report()
        report.extend(titles)
        report.append('By Sale Date Year')
        append_units(report)

        years = (2003, 2004, 2005, 2006, 2007, 2008, 2009)
        num_years = len(years)

        # build the rank table, which has the rankings of each predictor by year
        rank = {}  # key = (year, predictor)
        for year in years:
            nz_count = {}  # key = predictor
            for predictor in predictors_ordered:
                c = counts.by_predictor_year[(predictor, year)]
                nz_count[predictor] = c
            # produce list of tuples sorted by value
            s = sorted(nz_count.items(),
                       key=operator.itemgetter(1),
                       reverse=True)
            rank_number = 1
            for predictor, _ in s:
                rank[(year, predictor)] = rank_number
                rank_number += 1

        # print the rank table
        format_header = '%25s' + (' %4d' * num_years) + ' %4s'
        format_data = '%25s' + (' %4d' * num_years) + ' %4d'

        report.append(format_header % ('predictor',
                                       2003, 2004, 2005, 2006, 2007, 2008, 2009,
                                       'all'))
        overall_rank = 1
        for predictor in predictors_ordered:
            report.append(format_data % (predictor,
                                         rank[(2003, predictor)],
                                         rank[(2004, predictor)],
                                         rank[(2005, predictor)],
                                         rank[(2006, predictor)],
                                         rank[(2007, predictor)],
                                         rank[(2008, predictor)],
                                         rank[(2009, predictor)],
                                         overall_rank))
            overall_rank += 1
        report.write(txt_path('ranked-by-year'))

    def ranked_by_quarter(titles,
                          num_fitted_models,
                          predictors_ordered,
                          counts):
        report = Report()
        report.extend(titles)
        report.append('By Sale Date Quarter')
        append_units(report)

        quarters = (1, 2, 3, 4)
        num_quarters = len(quarters)

        # build the rank table, which has the rankings of each predictor by year
        rank = {}  # key = (quarter, predictor)
        for quarter in quarters:
            nz_count = {}  # key = predictor
            for predictor in predictors_ordered:
                c = counts.by_predictor_quarter[(predictor, quarter)]
                nz_count[predictor] = c
            # produce list of tuples sorted by value
            s = sorted(nz_count.items(),
                       key=operator.itemgetter(1),
                       reverse=True)
            rank_number = 1
            for predictor, _ in s:
                rank[(quarter, predictor)] = rank_number
                rank_number += 1

        # print the rank table
        format_header = '%25s' + ('   Q%1d' * num_quarters) + ' %4s'
        format_data = '%25s' + (' %4d' * num_quarters) + ' %4d'

        report.append(format_header % ('predictor',
                                       1, 2, 3, 4,
                                       'all'))
        overall_rank = 1
        for predictor in predictors_ordered:
            report.append(format_data % (predictor,
                                         rank[(1, predictor)],
                                         rank[(2, predictor)],
                                         rank[(3, predictor)],
                                         rank[(4, predictor)],
                                         overall_rank))
            overall_rank += 1
        report.write(txt_path('ranked-by-quarter'))

    def ranked_by_month(titles,
                        num_fitted_models,
                        predictors_ordered,
                        counts):
        report = Report()
        report.extend(titles)
        report.append('By Sale Date Month')
        append_units(report)

        month_names = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
        num_months = len(month_names)

        # build the rank table, which has the rankings of each predictor by year
        rank = {}  # key = (month, predictor)
        for month_index in xrange(len(month_names)):
            month_number = month_index + 1  # 1 ,2, ..., 12
            nz_count = {}  # key = predictor
            for predictor in predictors_ordered:
                c = counts.by_predictor_month[(predictor, month_number)]
                nz_count[predictor] = c
            # produce list of tuples sorted by value
            s = sorted(nz_count.items(),
                       key=operator.itemgetter(1),
                       reverse=True)
            rank_number = 1
            for predictor, _ in s:
                rank[(month_number, predictor)] = rank_number
                rank_number += 1

        # print the rank table
        format_header = '%25s' + (' %3s' * num_months) + ' %3s'
        format_data = '%25s' + (' %3d' * num_months) + ' %3d'

        def mn(i):
            return month_names[i]

        report.append(format_header % ('predictor',
                                       mn(0), mn(1), mn(2),
                                       mn(3), mn(4), mn(5),
                                       mn(6), mn(7), mn(8),
                                       mn(9), mn(10), mn(11),
                                       'all'))

        overall_rank = 1
        for predictor in predictors_ordered:
            def r(i):
                return rank[(i, predictor)]

            report.append(format_data % (predictor,
                                         r(1), r(2), r(3),
                                         r(4), r(5), r(6),
                                         r(7), r(8), r(9),
                                         r(10), r(11), r(12),
                                         overall_rank))
            overall_rank += 1
        report.write(txt_path('ranking-by-month'))

    def mean_coefficients_all_periods(num_tests, test_coef):
        # equally weight the days, so don't use num_tests
        # ARGS
        #  num_tests[(fold_number, sale_date)] -> count
        #  test_coef[(fold_number, sale_date, predictor_name)] -> coefficient
        pdb.set_trace()

        # sum coefficients and their abs values by predictor
        sum_absolute_coefficient = collections.defaultdict(int)
        sum_coefficient = collections.defaultdict(int)
        for k, coefficient in test_coef.iteritems():
            fold_number, sale_date, predictor_name = k
            sum_absolute_coefficient[predictor_name] += abs(coefficient)
            sum_coefficient[predictor_name] += coefficient

        # sort the predictor names
        print sum_absolute_coefficient
        print sum_coefficient
        pdb.set_trace()
        predictors = sorted(sum_absolute_coefficient.items(),
                            key=operator.itemgetter(1),  # sort by value
                            reverse=True)                # sort high first

        # print the report
        # coef | mean abs value | mean value
        pdb.set_trace()
        report = Report()
        report.append('Mean Values of Coefficient from the Lasso Regressions')
        report.append('For All Periods')
        report.append('Using Normalized Features')

        format_header = '%25s %11s %11s'
        format_data = '%25s %+11.6f %+11.6f'

        report.append(format_header % (' ', 'mean', ' '))
        report.append(format_header % (' ', 'absolute', 'mean'))
        report.append(format_header %
                      ('predictor', 'coefficient', 'coefficient'))

        num_models = 1.0 * len(test_coef)  # num items
        pdb.set_trace()
        for predictor_pair in predictors:
            predictor = predictor_pair[0]
            report.append(format_data %
                          (predictor,
                           sum_absolute_coefficient[predictor] / num_models,
                           sum_coefficient[predictor] / num_models))
        report.write(txt_path('mean-coefficients-all-periods'))

    # parse the reduced data
    num_tests = data['num_tests']
    test_coef = data['test_coef']
    del data

    num_fitted_models, predictors_ordered, counts = \
        make_counts(num_tests, test_coef)

    mean_coefficients_all_periods(num_tests, test_coef)
    title = 'Percent of All Models with Non-Zero Coefficients'
    nz_count_all_periods(title, num_fitted_models, predictors_ordered, counts)
    nz_count_by_year(title, num_fitted_models, predictors_ordered, counts)

    titles = ('Ranking for Inclusion of Predictors In Models',
              'Ranking = 1: Most Frequently Included Predictor')
    ranked_by_year(titles, num_fitted_models, predictors_ordered, counts)
    ranked_by_quarter(titles, num_fitted_models, predictors_ordered, counts)
    ranked_by_month(titles, num_fitted_models, predictors_ordered, counts)


def get_reduced_data(control):
    # return reduced data that corresponds to the --in PATH file

    def path_to_cache():
        return directory('working') + control.base_name + '.cache.pickle'

    def create_and_write_cache():
        print 'reading input file (slow)'
        reduction = create_reduction(control)
        print 'writing cache'
        f = open(path_to_cache(), 'wb')
        cache_record = {'in': control.arg_in, 'reduction': reduction}
        pickle.dump(cache_record, f)
        f.close()
        return reduction

    if control.arg_cache:
        # invocation specified --cache
        try:
            print 'attempting to read cache'
            f = open(path_to_cache(), 'rb')
        except IOError as e:
            if e[1] == 'No such file or directory':
                return create_and_write_cache()
            else:
                raise e
        cache_record = pickle.load(f)
        f.close()
        if cache_record['in'] == control.arg_in:
            return cache_record['reduction']
        else:
            return create_and_write_cache()
    else:
        return create_and_write_cache()


def main():
    control = make_control(sys.argv)
    path = \
        directory('log') + \
        control.base_name + '.' + control.now.isoformat('T') + '.log'
    sys.stdout = Logger(logfile_path=path)  # print x now logs and prints x
    print control

    reduced_data = get_reduced_data(control)
    create_charts(control, reduced_data)

    print control
    if control.testing:
        print 'DISCARD OUTPUT: TESTING'
    print 'done'

    return


if __name__ == '__main__':
    main()
