# create files (selective on how invoked)
# WORKING/chart-02.makefile
# WORKING/chart-02.data
# WORKING/chart-02.txt-mean-mae.txt
# WORKING/chart-02.txt-mean-rmeanse.txt
# WORKING/chart-02.txt-median-rmedianse.txt
# WORKING/chart-02.txt-median-medianae.txt

# import built-ins and libraries
import sys
import pdb
import cPickle as pickle
import numpy as np
import os.path

# import my stuff
from directory import directory
from Logger import Logger


def print_help():
    print 'python chart-02.py SUFFIX [ERROR]'
    print 'where SUFFIX in {"makefile", "data", "txt"}'
    print 'and   ERROR  in {"rmse", "mmae"}'


class Control(object):
    def __init__(self, arguments):
        self.me = 'chart-02'

        working = directory('working')
        log = directory('log')
        cells = directory('cells')
        src = directory('src')
        self.dir_working = working
        self.dir_log = log
        self.dir_cells = cells
        self.dir_src = src

        # handle command line arguments
        if len(arguments) not in (2, 3):
            print print_help()
            raise RuntimeError('missing command line argument')

        self.suffix = arguments[1]

        if len(arguments) == 3:
            self.error = arguments[2]
            if self.error == 'mean-root-mean-squared-errors':
                self.header = 'Mean of Root Mean Squared Errors'
            elif self.error == 'median-root-median-squared-errors':
                self.header = 'Median of Root Median Squared Errors'
            else:
                print print_help()
                raise RuntimeError('bad ERROR: ' + self.error)
            self.path_out_txt = working + self.me + '-' + self.error + '.txt'

        self.path_out_log = \
            log + self.me + '-' + '-'.join(arguments[1:]) + '.log'
        self.path_out_data = working + self.me + '.data'
        self.path_dir_cells = cells
        self.path_out_makefile = src + self.me + '.makefile'

        # components of cell names
        self.model = 'ols'
        self.responses = ['price', 'logprice']
        self.predictors = ['act', 'actlog', 'ct', 'ctlog']
        self.year = '2008'
        self.training_periods = ['30', '60', '90', '120', '150', '180',
                                 '210', '240', '270', '300', '330', '360']

        self.testing = False
        self.debugging = False

    def __str__(self):
        s = ''
        for kv in sorted(self.__dict__.items()):
            k, v = kv
            s = s + ('control.%s = %s\n' % (k, v))
        return s


class Report(object):

    def __init__(self, lines):
        self.lines = lines
        self.format_header = '{:>9s}' + (' {:>8s}' * 8)
        self.format_detail = '{:9d}' + (' {:8d}' * 8)

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


def create_txt(control):
    '''Return list of lines that are chart 02.txt.
    '''
    def append_description(lines):
        '''Append header lines'''
        lines.append(control.header)
        lines.append('From 10-fold Cross Validation')
        lines.append(' ')
        lines.append('Model: OLS')
        lines.append('Time period: 2008')
        lines.append(' ')

    def read_data():
        '''Return correct data dict built by create_data() function.'''
        f = open(control.path_out_data, 'rb')
        data = pickle.load(f)
        f.close()
        selected_data = data[control.error]
        return selected_data

    def append_header(t):
        t.header('response:',
                 'price', 'price', 'price', 'price',
                 'logprice', 'logprice', 'logprice', 'logprice')
        t.header('predForm:',
                 'level', 'level', 'log', 'log',
                 'level', 'level', 'log', 'log')
        t.header('use tax:',
                 'yes', 'no',
                 'yes', 'no',
                 'yes', 'no',
                 'yes', 'no')
        t.header('ndays', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ')

    def make_predictor(predForm, usetax):
        '''Return predictor.'''
        prefix = 'act' if usetax == 'yes' else 'ct'
        suffix = 'log' if predForm == 'log' else ''
        return prefix + suffix

    def append_detail_line(report, data, ndays):
        def v(response, predForm, usetax):
            '''Return int or 0, the value in the report.
            '''
            key = (response,
                   make_predictor(predForm, usetax),
                   ndays)
            if key in data:
                value = data[key]
                return int(value)
            else:
                print 'no data for', key
                return 0

        report.detail(int(ndays),
                      v('price', 'level', 'yes'),
                      v('price', 'level', 'no'),
                      v('price', 'log', 'yes'),
                      v('price', 'log', 'no'),
                      v('logprice', 'level', 'yes'),
                      v('logprice', 'level', 'no'),
                      v('logprice', 'log', 'yes'),
                      v('logprice', 'log', 'no'))

    def append_detail_lines(report, values):
        '''Append body lines to report using values.'''

        # one line for each training period
        for ndays in control.training_periods:
            append_detail_line(report, values, ndays)

    def write_lines(lines):
        f = open(control.path_out_txt, 'w')
        for line in lines:
            f.write(line)
            f.write('\n')
        f.close()

    lines = []
    append_description(lines)
    report = Report(lines)
    append_header(report)
    data = read_data()
    append_detail_lines(report, data)
    write_lines(lines)


def create_data(control):
    '''Write data file (in pickle format) to working directory.

    The data is a dict
    key = ERROR (from command line) (one of mrmse mae)
    value = a dicionary with the error
     key =(response, predictor, training_days)
     value = scalar value from each fold
    '''
    def make_file_path(response, predictor, training_days, control):
        '''Return string containing file name.
        '''
        cell_file_name = '%s-%s-%s-%s-%s.cvcell' % (control.model,
                                                    response,
                                                    predictor,
                                                    control.year,
                                                    training_days)
        return control.dir_cells + cell_file_name

    def get_cv_result(file_path):
        '''Return CvResult instance.'''
        f = open(file_path, 'rb')
        (cv_result, cv_cell_control) = pickle.load(f)
        f.close()
        return cv_result

    mean_rMeanSE = {}
    median_rMedianSE = {}

    # create table containing results from each cross validation
    for response in control.responses:
        for predictor in control.predictors:
            for training_period in control.training_periods:
                file_path = make_file_path(response,
                                           predictor,
                                           training_period,
                                           control)
                key = (response, predictor, training_period)
                if os.path.isfile(file_path):
                    cv_result = get_cv_result(file_path)

                    def save(d, method):
                        '''Save cv_result.method() into d[key].'''
                        value = method()
                        if value.has_value:
                            d[key] = value.value
                        else:
                            print 'no value for', key
                            d[key] = None

                    print file_path
                    save(mean_rMeanSE,
                         cv_result.mean_of_root_mean_squared_errors)
                    save(median_rMedianSE,
                         cv_result.median_of_root_median_squared_errors)
                else:
                    print 'no file for', response, predictor, training_period
                    raise RuntimeError('missing file: ' + file_path)

    # write the data
    data = {'mean-root-mean-squared-errors': mean_rMeanSE,
            'median-root-median-squared-errors': median_rMedianSE}
    print 'data'
    for k, v in data.iteritems():
        print k, v

    f = open(control.path_out_data, 'wb')
    pickle.dump(data, f)
    f.close()


def create_makefile(control):
    '''Write makefile to source directory.'''

    def make_file_names():
        '''Return list of cell names.'''
        file_names = []
        for response in control.responses:
            for predictor in control.predictors:
                for training_period in control.training_periods:
                    cell_name = '%s-%s-%s-%s-%s' % (control.model,
                                                    response,
                                                    predictor,
                                                    control.year,
                                                    training_period)
                    file_name = '%s%s.cvcell' % (control.dir_cells,
                                                 cell_name)
                    file_names.append(file_name)
                    if control.testing and len(file_names) > 0:
                        return file_names
        return file_names

    def make_lines():
        '''Produce lines for makefile.

        chart-02-cells = <cell1> <cell2> ...
        chart-02.txt: chart-02.data chart-02.py
            $(PYTHON) chart-02.py txt
        chart-02.data: $(chart-02-cells) chart-02.py
            $(PYTHON) chart-02.py data
        #chart-02.makefile: chart02.py
        #   $(PYTHON) chart-02.py makefile
        '''
        lines = []
        lines.append('%s-cells = %s' % (control.me,
                                        ' '.join(make_file_names())))

        lines.append('%s%s.txt: %s%s.data %s.py' % (control.dir_working,
                                                    control.me,
                                                    control.dir_working,
                                                    control.me,
                                                    control.me))
        lines.append('\t$(PYTHON) %s.py txt' % control.me)

        lines.append('%s%s.data: $(%s-cells) %s.py' % (control.dir_working,
                                                       control.me,
                                                       control.me,
                                                       control.me))
        lines.append('\t$(PYTHON) %s.py data' % control.me)

#        lines.append('%s.makefile: %s.py' % (control.me,
#                                             control.me))
#        lines.append('\t$(PYTHON) %s.py makefile' % control.me)
        return lines

    lines = make_lines()
    f = open(control.path_makefile, 'w')
    for line in lines:
        f.write(line)
        f.write('\n')
    f.close()


def main():

    control = Control(sys.argv)
    sys.stdout = Logger(logfile_path=control.path_out_log)
    print control

    if control.suffix == 'makefile':
        create_makefile(control)
    elif control.suffix == 'data':
        create_data(control)
    elif control.suffix == 'txt':
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
