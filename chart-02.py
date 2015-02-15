# create files
# WORKING/chart-02.makefile
# WORKING/chart-02.data
# WORKING/chart-02.txt     containing chart median losses across fold

# import built-ins and libraries
import sys
import pdb
import cPickle as pickle
import numpy as np
import os.path

# import my stuff
from directory import directory
from Logger import Logger
import Maybe


def print_help():
    print 'python chart-02.py SUFFIX'
    print 'where SUFFIX in {"makefile", "data", "txt"}'


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
        if len(arguments) != 2:
            print print_help()
            raise RuntimeError('missing command line argument')

        self.suffix = arguments[1]

        self.path_out_log = log + self.me + '.log'
        self.path_dir_cells = cells
        self.path_data = working + self.me + '.data'
        self.path_txt = working + self.me + '.txt'
        self.path_makefile = src + self.me + '.makefile'

        # components of cell names
        self.model = 'ols'
        self.responses = ['price', 'logprice']
        self.predictors = ['act', 'actlog', 'ct', 'ctlog']
        self.year = '2008'
        self.training_periods = ['30', '60', '90', '120', '150', '180',
                                 '210', '240', '270', '300', '330', '360']

        self.testing = False
        self.debugging = False


class Table(object):

    def __init__(self, lines):
        self.lines = lines
        self.format_header = '{:>9s}' + (' {:>8s}' * 8)
        self.format_detail = '{:9d}' + (' {:8d}' * 8)

    def header(self, c0, c1, c2, c3, c4, c5, c6, c7, c8):
        print c0, c1, c2, c3, c4, c5, c6, c7, c8
        s = self.format_header.format(c0, c1, c2, c3, c4, c5, c6, c7, c8)
        self.lines.append(s)

    def detail(self, ndays, c1, c2, c3, c4, c5, c6, c7, c8):
        print ndays, c1, c2, c3, c4, c5, c6, c7, c8
        s = self.format_detail.format(ndays, c1, c2, c3, c4, c5, c6, c7, c8)
        self.lines.append(s)


def create_txt(control):
    '''Return list of lines that are chart 02.
    '''
    def append_description(lines):
        '''Append header lines'''
        lines.append('Median of Root Median Squared Errors')
        lines.append('From 10-fold Cross Validation')
        lines.append(' ')
        lines.append('Model: OLS')
        lines.append('Time period: 2008')
        lines.append(' ')

    def read_values():
        '''Return table dict built by make_data.'''
        f = open(control.path_data, 'rb')
        values = pickle.load(f)
        f.close()
        return values

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

    def append_detail_lines(table, values):
        '''Append body lines to Table t using values.'''
        def append_line(ndays):
            def v(response, predForm, usetax):
                '''Return int or 0, the value in the table'''
                def make_predictor(predForm, usetax):
                    prefix = 'act' if usetax == 'yes' else 'ac'
                    suffix = 'log' if predForm == 'log' else ''
                    return prefix + suffix

                key = (response,
                       make_predictor(predForm, usetax),
                       ndays)
                if key in values:
                    potential_value = values[key]
                    if potential_value is None:
                        print 'None value for key', key
                        return 0
                    else:
                        return int(potential_value)
                else:
                    print 'no value for', key
                    return 0

            table.detail(int(ndays),
                         v('price', 'level', 'yes'),
                         v('price', 'level', 'no'),
                         v('price', 'log', 'yes'),
                         v('price', 'log', 'no'),
                         v('logprice', 'level', 'yes'),
                         v('logprice', 'level', 'no'),
                         v('logprice', 'log', 'yes'),
                         v('logprice', 'log', 'no'))

        # one line for each training period
        for ndays in control.training_periods:
            append_line(ndays)

    def write_lines(lines):
        f = open(control.path_txt, 'w')
        for line in lines:
            f.write(line)
            f.write('\n')
        f.close()

    lines = []
    append_description(lines)
    table = Table(lines)
    append_header(table)
    append_detail_lines(table, read_values())
    write_lines(lines)


def create_data(control):
    '''Write data file (in pickle format) to working directory.

    The data is a table
    key =(response, predictor, training_days)
    value = np.array with median values from each fold
    '''
    def make_file_path(response, predictor, training_days, control):
        '''Return string containing file name.
        '''
        cell_file_name = '%s-%s-%s-%s-%s.pickle' % (control.model,
                                                    response,
                                                    predictor,
                                                    control.year,
                                                    training_days)
        return control.dir_cells + cell_file_name

    def make_value(file_path):
        '''Return Maybe(root_median_squared values from file).'''
        f = open(file_path, 'rb')
        (cv_result, cv_cell_control) = pickle.load(f)
        f.close()

        # process the cv_result
        vector = cv_result.median_errors_ignore_nans()
        if vector.has_value:
            median_errors = vector.value
            root_median_squared_errors = np.sqrt(median_errors * median_errors)
            return Maybe.Maybe(root_median_squared_errors)
        else:
            return Maybe.NoValue()

    table = {}

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
                    medians = make_value(file_path)
                    if medians.has_value:
                        median_of_medians = np.median(medians.value)
                        table[key] = median_of_medians
                    else:
                        print 'no value for', file_path
                        table[key] = None
                else:
                    print 'no file for', response, predictor, training_period
                    table[key] = None

    # write the table
    print 'table'
    for k, v in table.iteritems():
        print k, v

    f = open(control.path_data, 'wb')
    pickle.dump(table, f)
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

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    if control.suffix == 'makefile':
        create_makefile(control)
    elif control.suffix == 'data':
        create_data(control)
    elif control.suffix == 'txt':
        create_txt(control)
    else:
        print_help()
        raise RuntimeError('bad command SUFFIX')

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    if control.testing:
        print 'DISCARD OUTPUT: TESTING'

    print 'done'

if __name__ == '__main__':
    main()
