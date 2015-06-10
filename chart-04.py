# create files for chart-04: lassocv (lasso with cross validation)
# SRC/chart-NN.makefile
# WORKING/chart-NN.data
#   dict([fold_number, sale_date, feature_name]) = number_of_models?
# WORKING/chart-NN.SPECIFIC.txt
#  for now, SPECIFIC={all-periods}

# import built-ins and libraries
import collections
import cPickle as pickle
import os.path
import pdb
import sys

# import my stuff
from Bunch import Bunch
from directory import directory
from Logger import Logger


def print_help():
    print 'python chart-04.py WHAT_FILE [TXT_CHOICE]'
    print 'where WHAT_FILE  in {"makefile", "data", "txt"}'
    print 'where TXT_CHOICE in {"all-periods"}'


def make_control(argv):
    # return a Bunch

    if not(2 <= len(argv) <= 3):
        print_help()
        print 'argv', argv
        raise RuntimeError('bad invocation')

    # supply common conrol values
    b = Bunch(debugging=False,
              base_name=argv[0].split('.')[0],
              cvcell_id='lassocv-logprice-ct-2003on-30',
              me=argv[0],
              specific=argv[2] if len(argv) == 3 else '',
              testing=False,
              training_data='transactions-subset2-train.pickle',
              txt_choices=['all-periods'],  # all possible
              what_file=argv[1])

    return b


class Report(object):

    def __init__(self, lines, table_entry_format):
        self.lines = lines
        self.format_header = '{:>9s}' + (' {:>8s}' * 8)
        self.format_detail = '{:9d}' + ((' {:%s}' % table_entry_format) * 8)
        self.format_legend = '{:80s}'

    def header(self, c0, c1, c2, c3, c4, c5, c6, c7, c8):
        print c0, c1, c2, c3, c4, c5, c6, c7, c8
        s = self.format_header.format(c0, c1, c2, c3, c4, c5, c6, c7, c8)
        self.lines.append(s)

    def detail(self, ndays, *clist):
        # replace large column values with all 9's
        print ndays, clist
        large_value = 99999999
        capped = [x if x <= large_value else large_value
                  for x in clist]
        s = self.format_detail.format(ndays,
                                      capped[0],
                                      capped[1],
                                      capped[2],
                                      capped[3],
                                      capped[4],
                                      capped[5],
                                      capped[6],
                                      capped[7])
        # s = self.format_detail.format(ndays, c1, c2, c3, c4, c5, c6, c7, c8)
        self.lines.append(s)

    def legend(self, txt):
        print 'legend', txt
        s = self.format_legend.format(txt)
        self.lines.append(s)


def create_txt(control):
    '''Return list of lines that are chart 02.txt.
    '''
    def append_description(lines):
        '''Append header lines'''
        lines.append(control.specs.title)
        lines.append('From 10-fold Cross Validation')
        lines.append(' ')
        lines.append('Model: ' + control.specs.model)
        lines.append('Time period: ' + control.specs.year)
        lines.append(' ')

    def read_data():
        '''Return correct data dict built by create_data() function.'''
        path = control.path.dir_working + control.base_name + '.data'
        f = open(path, 'rb')
        data = pickle.load(f)
        f.close()
        return data

    def append_header(t):
        t.header('response:',
                 'price', 'price', 'price', 'price',
                 'logprice', 'logprice', 'logprice', 'logprice')
        t.header('features:',
                 control.specs.feature_sets[0],
                 control.specs.feature_sets[1],
                 control.specs.feature_sets[2],
                 control.specs.feature_sets[3],
                 control.specs.feature_sets[0],
                 control.specs.feature_sets[1],
                 control.specs.feature_sets[2],
                 control.specs.feature_sets[3])
        t.header('ndays', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ')

    def append_detail_line(report, data, ndays):
        def v(response, features):
            '''Return int or 0, the value in the report.
            '''
            # features := predictor
            # shortened, so save one column in the printout
            key = (response, features, ndays)
            if key in data:
                value = data[key]
                if control.specs.metric == 'mean-wi10':
                    return value
                elif control.specs.metric == 'mean-mean':
                    return int(value)
                elif control.specs.metric == 'median-median':
                    return int(value)
                else:
                    raise RuntimeError('unknown metric: ' +
                                       control.specs.metric)
            else:
                print 'no data for', key
                return 0

        report.detail(int(ndays),
                      v('price', control.specs.feature_sets[0]),
                      v('price', control.specs.feature_sets[1]),
                      v('price', control.specs.feature_sets[2]),
                      v('price', control.specs.feature_sets[3]),
                      v('logprice', control.specs.feature_sets[0]),
                      v('logprice', control.specs.feature_sets[1]),
                      v('logprice', control.specs.feature_sets[2]),
                      v('logprice', control.specs.feature_sets[3]))

    def append_detail_lines(report, values):
        '''Append body lines to report using values.'''

        # one line for each training period
        for ndays in control.specs.training_periods:
            append_detail_line(report, values, ndays)

    feature_set_desc = \
        dict(act='features derived from accessor, census, and taxroll data',
             actlog='like at, but size features in log domain',
             ct='features derived from census and taxroll data',
             ctlog='like ct, but size features in log domain',
             t='features derived from taxroll data',
             tlog='like t, but size features in log domain'
             )

    def append_legend_lines(report):
        # print legend describing features sets actually used
        def r(s):
            report.legend(s)

        r(' ')
        r('features set definitions')
        for feature_set in control.specs.feature_sets:
            r(feature_set + ': ' + feature_set_desc[feature_set])
        r(' ')

    def write_lines(lines):
        f = open(control.path.out_output, 'w')
        for line in lines:
            f.write(line)
            f.write('\n')
        f.close()

    lines = []
    append_description(lines)
    report = Report(lines, control.table_entry_format)
    append_header(report)
    data = read_data()
    append_detail_lines(report, data)
    append_legend_lines(report)
    write_lines(lines)


def create_data(control):
    '''Write data file (in pickle format) to working directory.

    The data is a dict
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

    path = directory('cells') + control.cvcell_id + '.cvcell'
    cv_result = get_cv_result(path)  # a CvResult instance
    num_folds = len(cv_result.fold_results)
    for fold_number in xrange(num_folds):
        fold_result = cv_result.fold_results[fold_number]
        fold_actuals = fold_result.actuals
        fold_estimates = fold_result.estimates
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

    # write the data
    pdb.set_trace()
    data = {'num_tests': num_tests, 'test_coef': test_coef}
    path = directory('working') + control.base_name + '.data'
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


def create_makefile(control):
    '''Write makefile to source directory.'''

    def recipe(command, options):
        result = command
        i = 0
        while (i < len(options)):
            result += ' ' + options[i]
            i += 1

        return result

    def rule(target, prerequisites, recipes):
        line = target + ':'
        for prerequesite in prerequisites:
            line += ' ' + prerequesite

        # append recipes, preceeding each with a tab character
        lines = [line]
        for recipe in recipes:
            lines.append('\t' + recipe)

        if True:
            for line in lines:
                print line

        return lines

    def make_lines():
        '''Produce lines for makefile.

        # makefile generate by command python PGM makefile"
        <cv-cell>: cv-cell.py <training-data>
            $(PYTHON) cv-cell.py <cv-cell-name>
        chart-04.SPECIFIC.txt: chart-04.data chart-04.py
            $(PYTHON) chart-04.py txt SPECIFIC
        chart-04.data: $(chart-04-cells) chart-04.py
            $(PYTHON) chart-04.py data
        #chart-04.makefile: chart04.py
            $(PYTHON) chart-04.py makefile
        '''

        lines = []
        start_python = '$(PYTHON)'

        # comment: how file was generated
        lines.append('# makefile generated by python %s makefile' %
                     control.me)

        # rule to build the makefile itself
        makefile = control.base_name + '.makefile'
        program = control.base_name + '.py'
        create_makefile = recipe(start_python,
                                 [program, 'makefile'])
        lines.extend(rule(makefile, [program], [create_makefile]))

        # rule to build the cross-validation cell
        the_cell = directory('cells') + control.cvcell_id + '.cvcell'
        training_data = directory('working') + control.training_data
        create_cell = recipe(start_python,
                             ['cv-cell.py', control.cvcell_id])
        # don't rebuilt the cell if this source code file changes
        lines.extend(rule(the_cell, [training_data], [create_cell]))

        # rule to build the data file
        the_data = directory('working') + control.base_name + '.data'
        create_data = recipe(start_python,
                             [program, 'data'])
        lines.extend(rule(the_data, [the_cell], [create_data]))

        # rule to build the txt files
        for txt_choice in control.txt_choices:
            path_to_txt_file = \
                directory('working') + \
                control.base_name + \
                '.' + txt_choice +\
                '.txt'
            create_txt = recipe(start_python,
                                [program, 'txt', txt_choice])
            lines.extend(rule(path_to_txt_file,
                              [program, the_data],
                              [create_txt]))

        # phony targets (for development)
        phony_data = 'chart-04.data'
        lines.extend(rule('.PHONY', [phony_data], []))
        lines.extend(rule(phony_data, [the_data], []))

        return lines

    lines = make_lines()
    if True:
        print 'makefile'
        for line in lines:
            print line

    path = control.base_name + '.makefile'
    f = open(path, 'w')
    for line in lines:
        f.write(line)
        f.write('\n')
    f.close()


def main():
    pdb.set_trace()
    control = make_control(sys.argv)
    path = \
        directory('log') + \
        control.base_name + '.' + control.what_file + '.log'
    sys.stdout = Logger(logfile_path=path)  # print x now logs and prints x
    print control

    if control.what_file == 'makefile':
        create_makefile(control)
    elif control.what_file == 'data':
        create_data(control)
    elif control.what_file == 'txt':
        create_txt(control)
    else:
        print_help()
        raise RuntimeError('bad command SUFFIX')

    # clean up
    print control
    if control.testing:
        print 'DISCARD OUTPUT: TESTING'
    print 'done'


if __name__ == '__main__':
    main()
